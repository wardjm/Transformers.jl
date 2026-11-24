# Download helpers for the dataset `DataDep`s, built on the `Downloads` stdlib.
#
# The Google Drive logic is adapted from Fetch.jl's `gdownload`
# (https://github.com/foldfelis/Fetch.jl, MIT License, Copyright (c) 2021 JingYu Ning and contributors),
# vendored here because Fetch.jl is unmaintained and pinned deprecated/outdated dependencies (JSON3, HTTP 1).

using Downloads
using Random: randstring

const PROGRESS_INTERVAL = 2.0 # seconds between progress reports

function _progress_reporter(url, dest)
    start = last = time()
    return function (total, now)
        t = time()
        t - last < PROGRESS_INTERVAL && return
        last = t
        elapsed = t - start
        @info("Downloading", source = url, dest,
              downloaded = Base.format_bytes(now),
              total = total > 0 ? Base.format_bytes(total) : "unknown",
              progress = total > 0 ? round(now / total; digits = 4) : missing,
              average_speed = Base.format_bytes(now / max(elapsed, eps())) * "/s")
    end
end

"""
    download_with_progress(url, filepath; headers = Pair{String, String}[])

Download `url` to `filepath`, periodically logging progress. Errors on an HTTP error status. Returns the
 `Downloads.Response`.
"""
function download_with_progress(url, filepath; headers = Pair{String, String}[])
    resp = open(filepath, "w") do io
        Downloads.request(url; output = io, headers, progress = _progress_reporter(url, filepath))
    end
    resp isa Downloads.Response || throw(resp)
    if resp.status >= 400
        rm(filepath; force = true)
        error("Failed to download $url: HTTP status $(resp.status) $(resp.message)")
    end
    return resp
end

const SPREADSHEET_PATTERN = "docs.google.com/spreadsheets"
const DRIVE_PATTERN = "drive.google.com/file/d"
const DOCS_PATTERN = "docs.google.com"

is_gsheet(url) = occursin(SPREADSHEET_PATTERN, url)
is_gfile(url) = occursin(DRIVE_PATTERN, url)
is_gdoc(url) = occursin(DOCS_PATTERN, url)

# Resolve short links (e.g. bit.ly) to the Google URL they point at.
function unshortlink(url)
    (is_gdoc(url) || is_gfile(url)) && return url
    resp = Downloads.request(url; method = "HEAD", throw = false)
    return resp isa Downloads.Response ? resp.url : url
end

function gsheet_handler(url; format = :csv)
    link, expo = splitdir(url)
    if startswith(expo, "edit") || isempty(expo)
        url = link * "/export?format=$format"
    elseif startswith(expo, "export")
        url = replace(url, r"format=([a-zA-Z]*)(.*)" => SubstitutionString("format=$format\\2"))
    end
    return url
end

function gfile_handler(url)
    # pattern of file path in google drive:
    # https://drive.google.com/file/d/<hash>/view?usp=sharing
    h = match(r"/file/d/([^\/]+)/", url)
    isnothing(h) && error("Can't find google drive file ID in the url: $url")
    return "https://docs.google.com/uc?export=download&id=$(h.captures[])"
end

function find_filename(content_disposition)
    m = match(r"filename=\"([^\"]+)\"", content_disposition)
    isnothing(m) && (m = match(r"filename\*=UTF-8''([^;]+)", content_disposition))
    if isnothing(m)
        filename = "gdrive_downloaded-$(randstring())"
        @warn "File name not found, use `$filename`"
        return filename
    end
    return basename(m.captures[])
end

function getheader(resp::Downloads.Response, name)
    name = lowercase(name)
    for (k, v) in resp.headers
        lowercase(k) == name && return v
    end
    return nothing
end

function download_gdrive(url, localdir)
    # skip the "can't scan this file for viruses" interstitial for large files
    occursin(r"confirm=", url) || (url = "$url&confirm=t")
    tmp = tempname(localdir; cleanup = false)
    resp = try
        download_with_progress(url, tmp)
    catch
        rm(tmp; force = true)
        rethrow()
    end
    content_disposition = getheader(resp, "Content-Disposition")
    if isnothing(content_disposition)
        rm(tmp; force = true)
        error("Google Drive did not return a file for $url (no Content-Disposition header). ",
              "The file may be private, removed, or over its download quota.")
    end
    filepath = joinpath(localdir, find_filename(content_disposition))
    mv(tmp, filepath; force = true)
    return filepath
end

"""
    gdownload(url, localdir)

Download file or Google Sheet from Google drive into `localdir`. Returns the path of the downloaded file.
"""
function gdownload(url, localdir)
    url = unshortlink(url)
    if is_gfile(url)
        url = gfile_handler(url)
    elseif is_gsheet(url)
        url = gsheet_handler(url)
    end
    is_gdoc(url) || error("invalid Google Drive url: $url")
    return download_gdrive(url, localdir)
end
