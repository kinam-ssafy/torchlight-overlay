//! Korean titles share Tartarus's bundled PP-OCR model. Windows OCR is a secondary reader,
//! and remains the default for English. Only exact pixels reuse a cached result.

use vorax_scan::pixels::Img;
use windows::core::HSTRING;
use windows::Globalization::Language;
use windows::Graphics::Imaging::{BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap};
use windows::Media::Ocr::OcrEngine;
use windows::Storage::Streams::DataWriter;
use std::io::{Read, Write};
use std::os::windows::process::CommandExt;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct SharedReader {
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
}

impl SharedReader {
    fn start() -> std::io::Result<Self> {
        let exe = std::env::var_os("VORAX_OCR_HELPER").map(std::path::PathBuf::from)
            .unwrap_or(std::env::current_exe()?.with_file_name("TorchlightHelper.exe"));
        let mut child = Command::new(exe).arg("--ocr-stdio")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
            .creation_flags(0x08000000).spawn()?; // CREATE_NO_WINDOW
        Ok(Self { input: child.stdin.take().unwrap(), output: child.stdout.take().unwrap(), child })
    }

    fn read(&mut self, img: &Img) -> std::io::Result<String> {
        self.input.write_all(&(img.width as u32).to_le_bytes())?;
        self.input.write_all(&(img.height as u32).to_le_bytes())?;
        self.input.write_all(&img.data)?;
        self.input.flush()?;
        let mut length = [0; 4];
        self.output.read_exact(&mut length)?;
        let length = u32::from_le_bytes(length) as usize;
        if length > 65536 { return Err(std::io::Error::other("invalid OCR response length")); }
        let mut bytes = vec![0; length];
        self.output.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(std::io::Error::other)
    }
}

impl Drop for SharedReader {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}

pub struct Ocr {
    engine: std::cell::RefCell<Option<OcrEngine>>,
    pub language: String,
    pub reader_name: &'static str,
    korean: bool,
    cache: std::cell::RefCell<std::collections::VecDeque<(Img, String)>>,
    shared: std::cell::RefCell<Option<SharedReader>>,
}

impl Ocr {
    pub fn new() -> windows::core::Result<Ocr> {
        let requested = std::env::var("VORAX_LANGUAGE").unwrap_or_else(|_| "ko".into());
        let korean = requested == "ko";
        let shared = if korean { SharedReader::start().ok() } else { None };
        let reader_name = if shared.is_some() { "Tartarus PP-OCR" } else { "Windows OCR" };
        let (engine, language) = if shared.is_some() { (None, "ko-KR".to_owned()) }
            else {
                let engine = Self::windows_engine(korean)?;
                let language = engine.RecognizerLanguage()?.LanguageTag()?.to_string();
                (Some(engine), language)
            };
        Ok(Ocr { engine: std::cell::RefCell::new(engine), language, reader_name, korean, cache: Default::default(), shared: std::cell::RefCell::new(shared) })
    }

    fn windows_engine(korean: bool) -> windows::core::Result<OcrEngine> {
        let language = Language::CreateLanguage(&HSTRING::from(if korean { "ko-KR" } else { "en-US" }))?;
        if OcrEngine::IsLanguageSupported(&language)? { OcrEngine::TryCreateFromLanguage(&language) }
        else { OcrEngine::TryCreateFromUserProfileLanguages() }
    }

    /// The text in a picture (dark text on white), line by line.
    pub fn read(&self, img: &Img) -> windows::core::Result<String> {
        // Exact pixels only: animated text or a changed card never reuses an old OCR result.
        if let Some((_, text)) = self.cache.borrow().iter().find(|(old, _)| old.width == img.width && old.height == img.height && old.data == img.data) {
            return Ok(text.clone());
        }
        let mut text = String::new();
        {
            let mut shared = self.shared.borrow_mut();
            if let Some(reader) = shared.as_mut() {
                match reader.read(img) {
                    Ok(other) => text = other,
                    Err(_) => *shared = None,
                }
            }
        }
        if !vorax_scan::titles::confident_title(&text) {
            if let Ok(other) = self.read_windows(img) {
                if text.is_empty() || vorax_scan::titles::confident_title(&other) { text = other; }
            }
        }
        let mut cache = self.cache.borrow_mut();
        if cache.len() == 8 { cache.pop_front(); }
        cache.push_back((img.clone(), text.clone()));
        Ok(text)
    }

    fn read_windows(&self, img: &Img) -> windows::core::Result<String> {
        let mut engine = self.engine.borrow_mut();
        if engine.is_none() { *engine = Some(Self::windows_engine(self.korean)?); }
        let bmp = SoftwareBitmap::CreateWithAlpha(BitmapPixelFormat::Bgra8, img.width as i32, img.height as i32, BitmapAlphaMode::Premultiplied)?;
        let mut bgra = img.data.clone();
        for p in bgra.chunks_exact_mut(4) {
            p.swap(0, 2);
        }
        let writer = DataWriter::new()?;
        writer.WriteBytes(&bgra)?;
        bmp.CopyFromBuffer(&writer.DetachBuffer()?)?;
        let result = engine.as_ref().unwrap().RecognizeAsync(&bmp)?.get()?;
        let mut lines = Vec::new();
        for line in result.Lines()? {
            lines.push(line.Text()?.to_string());
        }
        Ok(lines.join("\n"))
    }
}
