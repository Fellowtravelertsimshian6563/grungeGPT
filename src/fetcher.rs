//! # Lyrics fetcher
//!
//! A model is only as interesting as the data it was trained on. The fetcher
//! exists so the project can bootstrap its own training corpus from the
//! internet without asking the user to hand-collect files. It downloads song
//! lyrics for a curated list of artists and saves them as plain text files
//! under `data/lyrics`.
//!
//! Finding lyrics is a two-step problem. First the fetcher asks the iTunes
//! Search API for candidate song titles for an artist. Then it asks lyrics.ovh
//! for the actual lyrics of each title. Real-world song listings are full of
//! noise, so the fetcher filters out titles containing words like "live",
//! "remaster", "instrumental", or "karaoke", which are not useful training
//! documents.
//!
//! The lyrics are cached locally and are deliberately not committed to git,
//! because licensing and repository size both matter.
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

/// iTunes Search API endpoint for song title discovery.
/// See <https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/>.
const ITUNES_SEARCH_URL: &str = "https://itunes.apple.com/search";

/// lyrics.ovh API endpoint for lyric text retrieval.
/// See <https://lyricsovh.docs.apiary.io>.
const LYRICS_OVH_URL: &str = "https://api.lyrics.ovh/v1";

/// Maximum song results to request from the iTunes search API per artist.
const ITUNES_SEARCH_LIMIT: u32 = 200;

/// Rate limit delay between consecutive lyrics API requests.
const REQUEST_DELAY_MS: u64 = 250;

/// Artists and directory slugs used by the lyrics downloader.
///
/// Each entry is `(display name, directory slug)`. The list spans grunge,
/// alternative rock, classic rock, and metal to give the model a broad
/// vocabulary of lyrical styles and themes.
pub const ARTISTS: &[(&str, &str)] = &[
    // Grunge
    ("Nirvana", "nirvana"),
    ("Alice in Chains", "alice-in-chains"),
    ("Pearl Jam", "pearl-jam"),
    ("Soundgarden", "soundgarden"),
    ("Stone Temple Pilots", "stone-temple-pilots"),
    ("Bush", "bush"),
    ("Silverchair", "silverchair"),
    // Alternative rock
    ("Smashing Pumpkins", "smashing-pumpkins"),
    ("Radiohead", "radiohead"),
    ("Foo Fighters", "foo-fighters"),
    ("Red Hot Chili Peppers", "red-hot-chili-peppers"),
    ("Nine Inch Nails", "nine-inch-nails"),
    ("Tool", "tool"),
    ("Audioslave", "audioslave"),
    // Classic rock
    ("The Beatles", "the-beatles"),
    ("The Rolling Stones", "the-rolling-stones"),
    ("Led Zeppelin", "led-zeppelin"),
    ("Pink Floyd", "pink-floyd"),
    ("Oasis", "oasis"),
    ("Guns N' Roses", "guns-n-roses"),
    ("Eagles", "eagles"),
    ("Fleetwood Mac", "fleetwood-mac"),
    ("The Doors", "the-doors"),
    ("Black Sabbath", "black-sabbath"),
    // Hard rock and metal
    ("Metallica", "metallica"),
    ("Seether", "seether"),
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
    let written: Vec<PathBuf> = ARTISTS
        .iter()
        .filter_map(|(artist, slug)| {
            match fetch_artist_lyrics(out_dir, artist, slug, max_songs) {
                Ok(path) => Some(path),
                Err(error) => {
                    eprintln!("=== {artist}: skipped ({error})");
                    None
                }
            }
        })
        .collect();
    Ok(written)
}

/// Download lyrics for a single artist and write them to a file.
///
/// # Parameters
///
/// - `out_dir`: root directory for artist subdirectories.
/// - `artist`: display name used for API queries.
/// - `slug`: directory and file name slug.
/// - `max_songs`: maximum lyrics to save.
///
/// # Returns
///
/// Path to the written lyric file.
///
/// # Errors
///
/// Returns an error when the directory or file cannot be created, or when
/// title fetching fails entirely.
fn fetch_artist_lyrics(
    out_dir: &Path,
    artist: &str,
    slug: &str,
    max_songs: usize,
) -> Result<PathBuf> {
    let artist_dir = out_dir.join(slug);
    std::fs::create_dir_all(&artist_dir)
        .with_context(|| format!("failed to create {}", artist_dir.display()))?;
    let output = artist_dir.join(format!("{slug}.txt"));
    println!("=== {artist}");

    let titles = fetch_titles(artist)?;
    println!("  found {} candidate titles", titles.len());

    let mut file = std::fs::File::create(&output)
        .with_context(|| format!("failed to create {}", output.display()))?;

    let count = titles
        .iter()
        .take(max_songs)
        .filter_map(|title| match fetch_lyrics_for(artist, title) {
            Ok(Some(lyrics)) => {
                let _ = writeln!(file, "# {title}\n\n{lyrics}\n");
                thread::sleep(Duration::from_millis(REQUEST_DELAY_MS));
                Some(())
            }
            Ok(None) => None,
            Err(error) => {
                eprintln!("  failed for {title}: {error}");
                None
            }
        })
        .count();

    println!("  wrote {count} songs to {}", output.display());
    Ok(output)
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
    let url = format!(
        "{ITUNES_SEARCH_URL}?term={encoded}&entity=song&limit={ITUNES_SEARCH_LIMIT}"
    );
    let body = ureq::get(&url)
        .call()
        .with_context(|| format!("iTunes request failed for {artist}"))?
        .into_string()
        .context("failed to read iTunes response")?;
    let json: Value = serde_json::from_str(&body).context("invalid iTunes JSON")?;

    let mut titles = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for result in json["results"].as_array().into_iter().flatten() {
        let Some(title) = result["trackName"].as_str().map(str::trim) else {
            continue;
        };
        if title.is_empty() || is_noise(title) {
            continue;
        }
        let key = title.to_lowercase();
        if seen.insert(key) {
            titles.push(title.to_string());
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
    let url = format!("{LYRICS_OVH_URL}/{artist}/{title}");
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
