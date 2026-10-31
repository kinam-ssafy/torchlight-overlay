using System.Diagnostics;
using System.IO;
using System.Net.Http;

namespace Tartarus.App;

/// <summary>Updates from our companion releases, never the upstream solver.</summary>
public static class Updates
{
    private static readonly HttpClient Http = CreateClient();
    private static HttpClient CreateClient()
    {
        var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("TorchlightOverlay/" + UpdatePackage.VersionNumber);
        http.DefaultRequestHeaders.Accept.ParseAdd("application/vnd.github+json");
        return http;
    }

    private static string AppDir => Path.TrimEndingDirectorySeparator(Path.GetFullPath(AppContext.BaseDirectory));
    private static string Staging => Path.Combine(AppDir, "user-data", "updates");
    public static bool Installed => File.Exists(Path.Combine(AppDir, "README.ko.md")) && File.Exists(Path.Combine(AppDir, "manifest.json"));

    public static async Task<UpdatePackage.Release?> LatestAsync(CancellationToken ct = default)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(15));
        return UpdatePackage.ParseRelease(await Http.GetStringAsync($"https://api.github.com/repos/{UpdatePackage.Repository}/releases/latest", timeout.Token));
    }

    public static async Task<string> DownloadAsync(UpdatePackage.Release release, IProgress<double>? progress, CancellationToken ct)
    {
        foreach (var dir in new[] { AppDir, Path.Combine(AppDir, "user-data"), Staging })
            if (Directory.Exists(dir) && (File.GetAttributes(dir) & FileAttributes.ReparsePoint) != 0) throw new IOException("업데이트 폴더에 연결 지점이 있습니다.");
        CleanUp();
        var session = Path.Combine(Staging, Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Staging);
        var zip = UpdatePackage.FileUnder(Staging, Path.GetFileName(session) + "/download.zip");
        Directory.CreateDirectory(session);
        try
        {
            await UpdatePackage.DownloadAsync(Http, release, zip, progress, ct);
            var app = await Task.Run(() => UpdatePackage.Extract(zip, Path.Combine(session, "unpacked"), ct), ct);
            File.Delete(zip);
            return app;
        }
        catch { DeleteSession(session); throw; }
    }

    public static void StartInstall(string app)
    {
        if (!Path.GetFullPath(app).StartsWith(Staging + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("업데이트 임시 경로가 아닙니다.");
        var start = new ProcessStartInfo(Path.Combine(app, UpdatePackage.Executable)) { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = app };
        start.ArgumentList.Add("--install-update");
        start.ArgumentList.Add(AppDir);
        start.ArgumentList.Add(Environment.ProcessId.ToString());
        if (Process.Start(start) == null) throw new IOException("업데이트 설치 프로그램을 시작하지 못했습니다.");
    }

    public static void InstallHere(string[] args)
    {
        string target = Path.GetFullPath(args[0]);
        try
        {
            var prefix = Path.Combine(target, "user-data", "updates") + Path.DirectorySeparatorChar;
            if (!AppDir.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) || !File.Exists(UpdatePackage.FileUnder(target, UpdatePackage.Executable)) || !File.Exists(UpdatePackage.FileUnder(target, "README.ko.md")))
                throw new InvalidDataException("업데이트 대상 폴더를 확인할 수 없습니다.");
            try
            {
                using var old = Process.GetProcessById(int.Parse(args[1]));
                if (!old.HasExited)
                {
                    if (!string.Equals(old.MainModule?.FileName, Path.Combine(target, UpdatePackage.Executable), StringComparison.OrdinalIgnoreCase)) throw new IOException("종료할 프로그램이 일치하지 않습니다.");
                    if (!old.WaitForExit(30_000)) throw new IOException("프로그램이 아직 종료되지 않았습니다. 업데이트를 다시 시도하세요.");
                }
            }
            catch (ArgumentException) { } // Already exited.
            var session = Directory.GetParent(Directory.GetParent(AppDir)!.FullName)!.FullName;
            UpdatePackage.Apply(AppDir, target, Path.Combine(session, "backup"));
            Write(target, "Updated to " + UpdatePackage.VersionNumber);
            Process.Start(new ProcessStartInfo(Path.Combine(target, UpdatePackage.Executable)) { UseShellExecute = false, WorkingDirectory = target });
        }
        catch (Exception ex)
        {
            Write(target, "Update failed: " + ex.Message);
            System.Windows.MessageBox.Show($"업데이트를 완료하지 못했습니다.\n{ex.Message}\n\n설치 파일: {AppDir}", "토치라이트 오버레이 업데이트", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Warning);
        }
    }

    // Keep recent backups; only our own GUID-named staging sessions can be removed.
    public static void CleanUp()
    {
        if (!Directory.Exists(Staging) || (File.GetAttributes(Staging) & FileAttributes.ReparsePoint) != 0) return;
        foreach (var dir in Directory.EnumerateDirectories(Staging))
            if (Directory.GetLastWriteTimeUtc(dir) < DateTime.UtcNow.AddDays(-7)) DeleteSession(dir);
    }

    private static void DeleteSession(string session)
    {
        var full = Path.GetFullPath(session);
        if (!Guid.TryParseExact(Path.GetFileName(full), "N", out _) || !string.Equals(Path.GetDirectoryName(full), Staging, StringComparison.OrdinalIgnoreCase)) return;
        if (!Directory.Exists(full) || (File.GetAttributes(full) & FileAttributes.ReparsePoint) != 0) return;
        try
        {
            var todo = new Stack<string>();
            todo.Push(full);
            while (todo.TryPop(out var directory))
                foreach (var entry in Directory.EnumerateFileSystemEntries(directory))
                {
                    if ((File.GetAttributes(entry) & FileAttributes.ReparsePoint) != 0) return;
                    if (Directory.Exists(entry)) todo.Push(entry);
                }
            Directory.Delete(full, recursive: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    private static void Write(string target, string message)
    {
        try
        {
            var dir = Path.Combine(target, "user-data", "tartarus");
            Directory.CreateDirectory(dir);
            File.AppendAllText(Path.Combine(dir, "app.log"), $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} {message}{Environment.NewLine}");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }
}
