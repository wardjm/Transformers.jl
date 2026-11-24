using DelimitedFiles

import ..Datasets: dataset, datafile, reader, Mode, Test, download_with_progress

diagnostic_init() = register(DataDep(
    "GLUE-Diagnostic",
    """
    GLUE Diagnostics Datasets
    """,
    "https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D",
    "0e13510b1bb14436ff7e2ee82338f0efb0133ecf2e73507a697dc210db3f05fd";
    fetch_method = (url, localdir) -> begin
        filepath = joinpath(localdir, "diagnostic.tsv")
        download_with_progress(url, filepath)
        filepath
    end
))

struct Diagnostic <: Dataset end

function testfile(::Diagnostic)
    sets, header = readdlm(datadep"GLUE-Diagnostic/diagnostic.tsv", '\t', String; header=true)
    [selectdim(sets, 2, i) for i = (2,3)]
end
