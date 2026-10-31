using System.Diagnostics;
using System.Windows;

namespace Tartarus.App;

public partial class ControlWindow
{
    private UpdatePackage.Release? _update;
    private readonly CancellationTokenSource _updateLifetime = new();
    private CancellationTokenSource? _updateDownload;
    private bool _checkingUpdate;
    private bool _updating;
    private bool _updatesReady;

    private void SetupUpdates()
    {
        UpdateCheck.IsChecked = _settings.CheckUpdates;
        UpdateStatus.Text = "버튼을 눌러 새 버전을 확인하세요.";
        _updatesReady = true;
        if (_settings.CheckUpdates && !PreviewMode)
            Loaded += async (_, _) => await CheckLatestAsync();
    }

    private void UpdateCheck_Changed(object sender, RoutedEventArgs e)
    {
        if (!_updatesReady) return;
        _settings.CheckUpdates = UpdateCheck.IsChecked == true;
        _settings.Save();
    }

    private async void CheckUpdateButton_Click(object sender, RoutedEventArgs e) => await CheckLatestAsync();

    private async Task CheckLatestAsync()
    {
        if (_checkingUpdate || _updating) return;
        _checkingUpdate = true;
        CheckUpdateButton.IsEnabled = false;
        UpdateStatus.Text = "새 버전 확인 중...";
        try
        {
            _update = await Updates.LatestAsync(_updateLifetime.Token);
            UpdateButton.Visibility = Visibility.Collapsed;
            UpdateNotesButton.Visibility = Visibility.Collapsed;
            if (_update == null) UpdateStatus.Text = "확인할 수 있는 정식 릴리스가 없습니다.";
            else if (_update.Version > UpdatePackage.Current)
            {
                UpdateStatus.Text = $"새 버전 {_update.Version}을 사용할 수 있습니다.";
                UpdateButton.Content = _update.Url != null && Updates.Installed ? "다운로드 및 업데이트" : "다운로드 페이지";
                UpdateButton.Visibility = Visibility.Visible;
                UpdateNotesButton.Visibility = Visibility.Visible;
                if (_update.Url == null) UpdateStatus.Text += " 자동 설치용 파일은 아직 준비되지 않았습니다.";
            }
            else UpdateStatus.Text = _update.Version == UpdatePackage.Current ? "최신 버전입니다." : $"공개 최신 버전은 {_update.Version}입니다. 현재 버전이 더 새롭습니다.";
        }
        catch (OperationCanceledException) when (_updateLifetime.IsCancellationRequested) { }
        catch (Exception ex)
        {
            UpdateStatus.Text = "업데이트 확인 실패. 네트워크 연결을 확인하고 다시 시도하세요.";
            UpdateStatus.ToolTip = ex.Message;
            Log("Companion update check failed: " + ex.Message);
        }
        finally { _checkingUpdate = false; CheckUpdateButton.IsEnabled = true; }
    }

    private void UpdateNotesButton_Click(object sender, RoutedEventArgs e)
    {
        if (_update is not { } release) return;
        var notes = release.Notes.Length > 3500 ? release.Notes[..3500] + "\n(전체 내용은 릴리스 페이지에서 확인하세요.)" : release.Notes;
        MessageBox.Show(this, string.IsNullOrWhiteSpace(notes) ? "변경 내용이 등록되지 않았습니다." : notes, $"토치라이트 오버레이 {release.Version} 변경 내용", MessageBoxButton.OK, MessageBoxImage.Information);
    }

    private async void UpdateButton_Click(object sender, RoutedEventArgs e)
    {
        if (_update is not { } release || _updating) return;
        if (release.Url == null || !Updates.Installed)
        {
            Process.Start(new ProcessStartInfo(release.Page) { UseShellExecute = true });
            return;
        }
        if (MessageBox.Show(this, $"버전 {release.Version} ({release.Size / 1048576.0:0} MB)을 다운로드하고 설치할까요?\n\n펫 단계와 설정, 로그는 유지됩니다. 설치할 때 오버레이가 종료되고 다시 실행됩니다.",
                "토치라이트 오버레이 업데이트", MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes) return;
        _updating = true;
        UpdateButton.IsEnabled = CheckUpdateButton.IsEnabled = false;
        CancelUpdateButton.Visibility = Visibility.Visible;
        using var download = CancellationTokenSource.CreateLinkedTokenSource(_updateLifetime.Token);
        _updateDownload = download;
        try
        {
            var progress = new Progress<double>(p => UpdateStatus.Text = $"업데이트 다운로드 중 {p * 100:0}%");
            string app = await Updates.DownloadAsync(release, progress, download.Token);
            download.Token.ThrowIfCancellationRequested();
            CancelUpdateButton.Visibility = Visibility.Collapsed;
            UpdateStatus.Text = "검증 완료. 업데이트 후 다시 시작합니다.";
            await Task.Run(() => Updates.StartInstall(app));
            Application.Current.Shutdown();
        }
        catch (OperationCanceledException) { UpdateStatus.Text = "다운로드를 취소했습니다. 현재 버전을 계속 사용할 수 있습니다."; }
        catch (Exception ex)
        {
            UpdateStatus.Text = "업데이트 실패. 현재 버전을 계속 사용할 수 있습니다.";
            Log("Companion update failed: " + ex.Message);
            MessageBox.Show(this, ex.Message, "토치라이트 오버레이 업데이트", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        finally
        {
            _updateDownload = null;
            _updating = false;
            UpdateButton.IsEnabled = CheckUpdateButton.IsEnabled = true;
            CancelUpdateButton.Visibility = Visibility.Collapsed;
        }
    }

    private void CancelUpdateButton_Click(object sender, RoutedEventArgs e) => _updateDownload?.Cancel();
}
