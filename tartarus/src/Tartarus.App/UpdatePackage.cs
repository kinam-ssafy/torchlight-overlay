using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text.Json;

namespace Tartarus.App;

/// <summary>The portable companion format, shared with deterministic updater tests.</summary>
public static class UpdatePackage
{
    public const string VersionNumber = "1.0.7";
    public const string Repository = "kinam-ssafy/torchlight-overlay";
    public const string FolderName = "TorchlightHelper-KO";
    public const string Executable = "TorchlightHelper.exe";
    public static Version Current => Version.Parse(VersionNumber);
    private const long MaxBytes = 2L * 1024 * 1024 * 1024;
    public sealed record Release(Version Version, string Page, string Notes, string? Asset, string? Url, long Size, string? Sha256);
    public sealed record Entry(string Path, long Bytes, string Sha256);

    public static Release? ParseRelease(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (root.GetProperty("draft").GetBoolean() || root.GetProperty("prerelease").GetBoolean()) return null;
        var tag = root.GetProperty("tag_name").GetString() ?? "";
        if (!Version.TryParse(tag.TrimStart('v'), out var version) || version.Build < 0 || version.Revision >= 0) return null;
        var page = $"https://github.com/{Repository}/releases/tag/{Uri.EscapeDataString(tag)}";
        var notes = root.TryGetProperty("body", out var body) ? body.GetString() ?? "" : "";
        var name = $"TorchlightHelper-KO-{version}-win-x64.zip";
        foreach (var a in root.GetProperty("assets").EnumerateArray())
        {
            if (a.GetProperty("name").GetString() != name) continue;
            var url = a.GetProperty("browser_download_url").GetString();
            var expected = $"https://github.com/{Repository}/releases/download/{Uri.EscapeDataString(tag)}/{name}";
            var digest = a.TryGetProperty("digest", out var d) ? d.GetString() : null;
            var size = a.GetProperty("size").GetInt64();
            if (url == expected && digest?.StartsWith("sha256:", StringComparison.Ordinal) == true && IsHash(digest[7..]) && size is > 0 and <= MaxBytes)
                return new Release(version, page, notes, name, url, size, digest[7..]);
        }
        return new Release(version, page, notes, null, null, 0, null);
    }

    private static bool IsHash(string s) => s.Length == 64 && s.All(Uri.IsHexDigit);
    public static string Hash(string file)
    {
        using var stream = File.OpenRead(file);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    /// <summary>A regular relative file, with no traversal, Windows aliases, or reparse points.</summary>
    public static string FileUnder(string root, string relative)
    {
        var parts = relative.Replace('\\', '/').Split('/');
        if (parts.Any(p => string.IsNullOrWhiteSpace(p) || p is "." or ".." || p.EndsWith('.') || p.EndsWith(' ') || p.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0))
            throw new InvalidDataException("업데이트 파일 경로가 올바르지 않습니다.");
        if (parts[0].Equals("user-data", StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("업데이트에 사용자 데이터가 포함되어 있습니다.");
        var current = Path.GetFullPath(root);
        CheckLink(current);
        foreach (var part in parts) { current = Path.Combine(current, part); CheckLink(current); }
        return current;
    }

    private static void CheckLink(string path)
    {
        if ((File.Exists(path) || Directory.Exists(path)) && (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("업데이트 경로에 링크 또는 연결 지점이 있습니다.");
    }

    private static IEnumerable<string> Files(string root)
    {
        CheckLink(root);
        foreach (var path in Directory.EnumerateFileSystemEntries(root))
        {
            CheckLink(path);
            if (Directory.Exists(path)) { foreach (var file in Files(path)) yield return file; }
            else yield return path;
        }
    }

    public static async Task DownloadAsync(HttpClient http, Release release, string zip, IProgress<double>? progress, CancellationToken ct)
    {
        if (release.Url == null || release.Sha256 == null || !IsHash(release.Sha256)) throw new InvalidDataException("검증 가능한 배포 ZIP이 없습니다.");
        using var response = await http.GetAsync(release.Url, HttpCompletionOption.ResponseHeadersRead, ct);
        response.EnsureSuccessStatusCode();
        await using var source = await response.Content.ReadAsStreamAsync(ct);
        await using var file = new FileStream(zip, FileMode.CreateNew);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[65536];
        long done = 0;
        int n;
        while ((n = await source.ReadAsync(buffer, ct)) > 0)
        {
            done += n;
            if (done > release.Size || done > MaxBytes) throw new InvalidDataException("업데이트 파일 크기가 배포 정보와 다릅니다.");
            await file.WriteAsync(buffer.AsMemory(0, n), ct);
            hash.AppendData(buffer, 0, n);
            progress?.Report((double)done / release.Size);
        }
        if (done != release.Size || !Convert.ToHexString(hash.GetHashAndReset()).Equals(release.Sha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("업데이트 파일 검증에 실패했습니다. 기존 버전은 변경하지 않았습니다.");
    }

    public static string Extract(string zip, string destination, CancellationToken ct = default)
    {
        if (Directory.Exists(destination)) throw new IOException("업데이트 임시 폴더가 이미 있습니다.");
        using var archive = ZipFile.OpenRead(zip);
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        long total = 0;
        foreach (var entry in archive.Entries)
        {
            var name = entry.FullName.Replace('\\', '/');
            if (!name.StartsWith(FolderName + "/", StringComparison.Ordinal) || name.EndsWith('/') || !names.Add(name))
                throw new InvalidDataException("토치라이트 오버레이 배포 ZIP 형식이 아닙니다.");
            if (((entry.ExternalAttributes >> 16) & 0xF000) == 0xA000) throw new InvalidDataException("ZIP에 링크가 포함되어 있습니다.");
            FileUnder(Path.Combine(destination, FolderName), name[(FolderName.Length + 1)..]);
            total = checked(total + entry.Length);
            if (total > MaxBytes) throw new InvalidDataException("업데이트 압축 해제 크기가 너무 큽니다.");
        }
        foreach (var entry in archive.Entries)
        {
            ct.ThrowIfCancellationRequested();
            var path = FileUnder(destination, entry.FullName);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            entry.ExtractToFile(path);
        }
        var app = Path.Combine(destination, FolderName);
        Validate(app);
        return app;
    }

    public static IReadOnlyList<Entry> Validate(string app)
    {
        var manifest = FileUnder(app, "manifest.json");
        var entries = JsonSerializer.Deserialize<List<Entry>>(File.ReadAllText(manifest), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidDataException("배포 매니페스트가 없습니다.");
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "manifest.json" };
        foreach (var entry in entries)
        {
            var file = FileUnder(app, entry.Path);
            if (!paths.Add(entry.Path.Replace('\\', '/')) || !IsHash(entry.Sha256) || !File.Exists(file) || new FileInfo(file).Length != entry.Bytes || !Hash(file).Equals(entry.Sha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"배포 파일 검증 실패: {entry.Path}");
        }
        if (!paths.Contains(Executable) || !paths.Contains("vorax-overlay.exe") || !paths.Contains("README.ko.md")) throw new InvalidDataException("배포 실행 파일이 빠져 있습니다.");
        if (Files(app).Count() != paths.Count) throw new InvalidDataException("매니페스트에 없는 배포 파일이 있습니다.");
        return entries;
    }

    /// <summary>Only changed package files are replaced. User data and unrelated files are untouched.</summary>
    public static void Apply(string source, string target, string backup)
    {
        var entries = Validate(source).Append(new Entry("manifest.json", new FileInfo(FileUnder(source, "manifest.json")).Length, Hash(FileUnder(source, "manifest.json")))).ToList();
        var changed = entries.Where(e => !File.Exists(FileUnder(target, e.Path)) || !Hash(FileUnder(target, e.Path)).Equals(e.Sha256, StringComparison.OrdinalIgnoreCase)).ToList();
        var restored = new List<(string Target, string? Backup)>();
        foreach (var entry in changed)
        {
            var to = FileUnder(target, entry.Path);
            var saved = FileUnder(backup, entry.Path);
            if (File.Exists(to)) { Directory.CreateDirectory(Path.GetDirectoryName(saved)!); File.Copy(to, saved, overwrite: false); }
        }
        try
        {
            foreach (var entry in changed)
            {
                var to = FileUnder(target, entry.Path);
                var saved = FileUnder(backup, entry.Path);
                Directory.CreateDirectory(Path.GetDirectoryName(to)!);
                restored.Add((to, File.Exists(saved) ? saved : null));
                File.Copy(FileUnder(source, entry.Path), to, overwrite: true);
            }
        }
        catch (Exception installError)
        {
            var errors = new List<Exception> { installError };
            foreach (var (to, saved) in restored.AsEnumerable().Reverse())
            {
                try { if (saved != null) File.Copy(saved, to, overwrite: true); else if (File.Exists(to)) File.Delete(to); }
                catch (Exception rollbackError) { errors.Add(rollbackError); }
            }
            if (errors.Count > 1) throw new AggregateException($"업데이트 복구를 완료하지 못했습니다. 백업: {backup}", errors);
            throw new IOException("업데이트를 적용하지 못해 이전 파일로 복구했습니다.", installError);
        }
    }
}
