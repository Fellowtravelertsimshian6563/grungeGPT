//! # Lyrics fetcher
//!
//! Downloads song lyrics from the public lyrics.ovh API.
//!
//! This module exists so the project can bootstrap its own training data. The
//! lyrics are cached as plain text files under `data/lyrics` and are not
//! committed to git.
//!
//! ## API notes
//!
//! lyrics.ovh is a free, unofficial lyrics API. The fetcher requests one song
//! per artist, filters out boilerplate noise, and saves each result as a text
//! file.
//!
//! ## References
//!
//! - lyrics.ovh API: <https://lyricsovh.docs.apiary.io>
//! - For larger public corpora, Project Gutenberg: <https://www.gutenberg.org>

use anyhow::{Context, Result};
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

/// Artists and directory slugs used by the lyrics downloader.
///
/// Each entry is `(display name, directory slug)`.
pub const ARTISTS: &[(&str, &str)] = &[
    ("Nirvana", "nirvana"),
    ("Alice in Chains", "alice-in-chains"),
    ("Pearl Jam", "pearl-jam"),
    ("The Beatles", "the-beatles"),
    ("The Rolling Stones", "the-rolling-stones"),
    ("Oasis", "oasis"),
    ("Led Zeppelin", "led-zeppelin"),
    ("Eagles", "eagles"),
    ("Seether", "seether"),
    ("Metallica", "metallica"),
    ("Soundgarden", "soundgarden"),
    ("Audioslave", "audioslave"),
    ("Guns N' Roses", "guns-n-roses"),
];

/// Download lyrics for all configured artists into `out_dir`.
///
/// # Parameters
///
/// - `out_dir`: destination directory for artist subdirectories.
/// - `max_songs`: maximum number of lyrics to save per artist.
///
/// # Returns
///
/// A list of paths to the created lyric files.
///
/// # Errors
///
/// Returns an error when an output directory cannot be created.
///
/// # Behavior
///
/// For each artist, song titles are fetched from the iTunes Search API and
/// lyrics are fetched from lyrics.ovh. Non-lyric noise titles are filtered out.
pub fn fetch_lyrics(out_dir: &Path, max_songs: usize) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    for (artist, slug) in ARTISTS {
        let artist_dir = out_dir.join(slug);
        std::fs::create_dir_all(&artist_dir)
            .with_context(|| format!("failed to create {}", artist_dir.display()))?;
        let output = artist_dir.join(format!("{slug}.txt"));
        println!("=== {artist}");

        let titles = match fetch_titles(artist) {
            Ok(titles) => titles,
            Err(error) => {
                eprintln!("  failed to fetch titles for {artist}: {error}");
                continue;
            }
        };
        println!("  found {} candidate titles", titles.len());

        let mut file = std::fs::File::create(&output)
            .with_context(|| format!("failed to create {}", output.display()))?;
        let mut count = 0usize;
        for title in titles {
            if count >= max_songs {
                break;
            }
            let lyrics = match fetch_lyrics_for(artist, &title) {
                Ok(Some(lyrics)) => lyrics,
                Ok(None) => continue,
                Err(error) => {
                    eprintln!("  failed for {title}: {error}");
                    continue;
                }
            };
            writeln!(file, "# {title}\n\n{lyrics}\n")?;
            count += 1;
            thread::sleep(Duration::from_millis(250));
        }
        println!("  wrote {count} songs to {}", output.display());
        written.push(output);
    }
    Ok(written)
}

/// Fetch candidate song titles for an artist from the iTunes Search API.
///
/// # Parameters
///
/// - `artist`: artist name to search for.
///
/// # Returns
///
/// A deduplicated list of title strings with noise titles removed.
///
/// # Errors
///
/// Returns an error when the iTunes request fails or the response is invalid.
fn fetch_titles(artist: &str) -> Result<Vec<String>> {
    let encoded = urlencoding::encode(artist);
    let url = format!("https://itunes.apple.com/search?term={encoded}&entity=song&limit=200");
    let body = ureq::get(&url)
        .call()
        .with_context(|| format!("iTunes request failed for {artist}"))?
        .into_string()
        .context("failed to read iTunes response")?;
    let json: Value = serde_json::from_str(&body).context("invalid iTunes JSON")?;

    let mut titles = Vec::new();
    let mut seen = std::collections::HashSet::new();
    if let Some(results) = json["results"].as_array() {
        for result in results {
            let Some(title) = result["trackName"].as_str() else {
                continue;
            };
            if is_noise(title) {
                continue;
            }
            let key = title.trim().to_lowercase();
            if !key.is_empty() && seen.insert(key) {
                titles.push(title.trim().to_string());
            }
        }
    }
    Ok(titles)
}

/// Fetch lyrics for a specific artist and title from lyrics.ovh.
///
/// # Parameters
///
/// - `artist`: artist name.
/// - `title`: song title.
///
/// # Returns
///
/// `Ok(Some(lyrics))` when found, `Ok(None)` when the API returns 404, and an
/// error for other failures.
fn fetch_lyrics_for(artist: &str, title: &str) -> Result<Option<String>> {
    let artist = urlencoding::encode(artist);
    let title = urlencoding::encode(title);
    let url = format!("https://api.lyrics.ovh/v1/{artist}/{title}");
    match ureq::get(&url).call() {
        Ok(response) => {
            let body = response
                .into_string()
                .context("failed to read lyrics response")?;
            let json: Value = serde_json::from_str(&body).context("invalid lyrics JSON")?;
            Ok(json["lyrics"].as_str().map(str::to_string))
        }
        Err(ureq::Error::Status(404, _)) => Ok(None),
        Err(error) => Err(error).context("lyrics request failed"),
    }
}

/// Return true when a title looks like a non-lyric release variant.
///
/// # Parameters
///
/// - `title`: song title to inspect.
///
/// # Returns
///
/// `true` when the title contains noise markers like "live", "remaster", or
/// "instrumental".
fn is_noise(title: &str) -> bool {
    let lower = title.to_lowercase();
    const NOISE: &[&str] = &[
        "live",
        "remaster",
        "demo",
        "instrumental",
        "karaoke",
        " mix",
        " edit",
        "version",
        "session",
        "acoustic",
        "anniversary",
        "deluxe",
        "bonus",
        "reprise",
        "mono",
        "stereo",
        " single",
        "radio",
        "alternate",
        "re-record",
        "reissue",
        " take",
    ];
    NOISE.iter().any(|word| lower.contains(word))
}
