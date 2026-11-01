using System.IO;
using System.Text;
using Tartarus.Vision;
using Tartarus.Vision.Ocr;

namespace Tartarus.App;

/// <summary>Local stdio worker sharing Tartarus's Korean OCR with the native card reader.</summary>
internal static class CardOcrServer
{
    public static void Run()
    {
        using var input = new BinaryReader(Console.OpenStandardInput());
        using var output = new BinaryWriter(Console.OpenStandardOutput());
        using var ocr = new OcrEngine(threads: 2, gameLanguage: "ko") { ReuseUnchanged = false, MinConf = 0.8f };
        while (true)
        {
            int width;
            try { width = input.ReadInt32(); }
            catch (EndOfStreamException) { return; }
            int height = input.ReadInt32();
            if (width is < 1 or > 2048 || height is < 1 or > 2048) return;
            int size = checked(width * height * 4);
            byte[] pixels = input.ReadBytes(size);
            if (pixels.Length != size) return;
            var frame = new Frame(width, height, pixels, PixelFormat.Rgba32);
            string text = string.Join("\n", ocr.Read(frame).Select(line => line.Text));
            byte[] encoded = Encoding.UTF8.GetBytes(text);
            output.Write(encoded.Length);
            output.Write(encoded);
            output.Flush();
        }
    }
}
