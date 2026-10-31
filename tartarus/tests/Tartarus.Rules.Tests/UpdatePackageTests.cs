using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Tartarus.App;

namespace Tartarus.Rules.Tests;

public sealed class UpdatePackageTests : IDisposable
{
    private readonly string _root = Path.Combine(AppContext.BaseDirectory, "updater-tests", Guid.NewGuid().ToString("N"));
    public UpdatePackageTests() => Directory.CreateDirectory(_root);
    private string PathAt(string name) => Path.Combine(_root, name);

    private static string ReleaseJson(string tag = "v1.10.0", string repo = UpdatePackage.Repository, string? digest = null, bool preview = false)
    {
        var name = $"TorchlightHelper-KO-{tag.TrimStart('v')}-win-x64.zip";
        return JsonSerializer.Serialize(new { tag_name = tag, draft = false, prerelease = preview, body = "한국어 변경 내용", assets = new[] {
            new { name, size = 123L, digest = digest ?? "sha256:" + new string('a', 64), browser_download_url = $"https://github.com/{repo}/releases/download/{tag}/{name}" }
        }});
    }

    [Fact]
    public void VersionsAreNumericAndOnlyCompanionAssetIsAccepted()
    {
        var release = UpdatePackage.ParseRelease(ReleaseJson())!;
        Assert.True(release.Version > new Version(1, 9, 0));
        Assert.NotNull(release.Url);
        Assert.Equal("한국어 변경 내용", release.Notes);
        Assert.Null(UpdatePackage.ParseRelease(ReleaseJson(repo: "glcoder/TartarusSolver"))!.Url);
    }

    [Theory]
    [InlineData("")]
    [InlineData("sha256:abcd")]
    [InlineData("md5:0123456789abcdef")]
    public void MissingOrInvalidDigestDisablesAutomaticInstall(string digest) => Assert.Null(UpdatePackage.ParseRelease(ReleaseJson(digest: digest))!.Url);

    [Theory]
    [InlineData("v1.2.0-beta")]
    [InlineData("v1.2")]
    [InlineData("v1.2.3.4")]
    public void NonReleaseVersionIsIgnored(string tag) => Assert.Null(UpdatePackage.ParseRelease(ReleaseJson(tag)));

    [Fact]
    public void PrereleaseIsIgnored() => Assert.Null(UpdatePackage.ParseRelease(ReleaseJson(preview: true)));

    private string Payload(string name, string content)
    {
        var app = PathAt(name);
        Directory.CreateDirectory(app);
        var files = new[] { UpdatePackage.Executable, "new-data.bin", "vorax-overlay.exe", "README.ko.md" };
        foreach (var file in files) File.WriteAllText(Path.Combine(app, file), content + ":" + file);
        File.WriteAllText(Path.Combine(app, "manifest.json"), JsonSerializer.Serialize(files.Select(file => new UpdatePackage.Entry(file, new FileInfo(Path.Combine(app, file)).Length, UpdatePackage.Hash(Path.Combine(app, file))))));
        return app;
    }

    private string Archive(string app, string? badEntry = null, bool link = false)
    {
        var path = PathAt(Guid.NewGuid().ToString("N") + ".zip");
        using var zip = ZipFile.Open(path, ZipArchiveMode.Create);
        foreach (var file in Directory.GetFiles(app)) zip.CreateEntryFromFile(file, UpdatePackage.FolderName + "/" + Path.GetFileName(file));
        if (badEntry != null)
        {
            var entry = zip.CreateEntry(badEntry);
            if (link) entry.ExternalAttributes = 0xA000 << 16;
            using var writer = new StreamWriter(entry.Open());
            writer.Write("bad");
        }
        return path;
    }

    [Fact]
    public void ExtractionAndInstallPreserveSettingsAndUnrelatedFiles()
    {
        var payload = Payload("payload", "new");
        var extracted = UpdatePackage.Extract(Archive(payload), PathAt("unpacked"));
        var target = Payload("install", "old");
        Directory.CreateDirectory(Path.Combine(target, "user-data", "tartarus"));
        var settings = Path.Combine(target, "user-data", "tartarus", "settings.json");
        File.WriteAllText(settings, "{\"ShelleyLevel\":5}");
        File.WriteAllText(Path.Combine(target, "personal.txt"), "keep");
        UpdatePackage.Apply(extracted, target, PathAt("backup"));
        Assert.Equal("new:" + UpdatePackage.Executable, File.ReadAllText(Path.Combine(target, UpdatePackage.Executable)));
        Assert.Equal("{\"ShelleyLevel\":5}", File.ReadAllText(settings));
        Assert.Equal("keep", File.ReadAllText(Path.Combine(target, "personal.txt")));
        Assert.Equal("old:" + UpdatePackage.Executable, File.ReadAllText(PathAt("backup/" + UpdatePackage.Executable)));
    }

    [Theory]
    [InlineData("TorchlightHelper-KO/../../outside.bin")]
    [InlineData("TorchlightHelper-KO/user-data/tartarus/settings.json")]
    [InlineData("TorchlightHelper-KO/README.ko.md")]
    [InlineData("TorchlightHelper-KO/unsafe:stream")]
    [InlineData("TartarusSolver/other.exe")]
    public void InvalidArchivesWriteNothing(string entry)
    {
        var zip = Archive(Payload("payload", "new"), entry);
        Assert.Throws<InvalidDataException>(() => UpdatePackage.Extract(zip, PathAt("unpacked")));
        Assert.False(Directory.Exists(PathAt("unpacked")));
        Assert.False(File.Exists(PathAt("outside.bin")));
    }

    [Fact]
    public void SymlinkArchiveIsRejected()
    {
        var zip = Archive(Payload("payload", "new"), "TorchlightHelper-KO/link", link: true);
        Assert.Throws<InvalidDataException>(() => UpdatePackage.Extract(zip, PathAt("unpacked")));
    }

    [Fact]
    public void TamperedPayloadDoesNotTouchInstallation()
    {
        var source = Payload("payload", "new");
        var target = Payload("install", "old");
        File.AppendAllText(Path.Combine(source, "vorax-overlay.exe"), "corrupt");
        Assert.Throws<InvalidDataException>(() => UpdatePackage.Apply(source, target, PathAt("backup")));
        Assert.Equal("old:" + UpdatePackage.Executable, File.ReadAllText(Path.Combine(target, UpdatePackage.Executable)));
        Assert.False(Directory.Exists(PathAt("backup")));
    }

    [Fact]
    public void CopyFailureRollsBackAlreadyReplacedAndNewFiles()
    {
        var source = Payload("payload", "new");
        var target = PathAt("install");
        Directory.CreateDirectory(target);
        File.WriteAllText(Path.Combine(target, UpdatePackage.Executable), "old executable");
        Directory.CreateDirectory(Path.Combine(target, "vorax-overlay.exe"));
        Assert.Throws<IOException>(() => UpdatePackage.Apply(source, target, PathAt("backup")));
        Assert.Equal("old executable", File.ReadAllText(Path.Combine(target, UpdatePackage.Executable)));
        Assert.False(File.Exists(Path.Combine(target, "new-data.bin")));
    }

    private sealed class ResponseHandler(byte[] bytes) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) });
        }
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(-1, false)]
    [InlineData(1, false)]
    [InlineData(0, true)]
    public async Task DownloadChecksExactSizeAndDigest(int sizeChange, bool wrongHash)
    {
        var bytes = Encoding.UTF8.GetBytes("release bytes");
        using var http = new HttpClient(new ResponseHandler(bytes));
        var release = new UpdatePackage.Release(new Version(1, 10, 0), "", "", "release.zip", "https://github.com/test", bytes.Length + sizeChange,
            wrongHash ? new string('0', 64) : Convert.ToHexString(SHA256.HashData(bytes)));
        if (sizeChange == 0 && !wrongHash)
        {
            await UpdatePackage.DownloadAsync(http, release, PathAt("download.zip"), null, CancellationToken.None);
            Assert.Equal(bytes, File.ReadAllBytes(PathAt("download.zip")));
        }
        else await Assert.ThrowsAsync<InvalidDataException>(() => UpdatePackage.DownloadAsync(http, release, PathAt("download.zip"), null, CancellationToken.None));
    }

    [Fact]
    public async Task CancelledDownloadDoesNotCreateAFile()
    {
        using var http = new HttpClient(new ResponseHandler([1, 2, 3]));
        var release = new UpdatePackage.Release(new Version(1, 10, 0), "", "", "release.zip", "https://github.com/test", 3, new string('a', 64));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => UpdatePackage.DownloadAsync(http, release, PathAt("download.zip"), null, new CancellationToken(true)));
        Assert.False(File.Exists(PathAt("download.zip")));
    }

    public void Dispose()
    {
        var boundary = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "updater-tests")) + Path.DirectorySeparatorChar;
        if (!Path.GetFullPath(_root).StartsWith(boundary, StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("Test path outside workspace");
        Directory.Delete(_root, recursive: true);
    }
}
