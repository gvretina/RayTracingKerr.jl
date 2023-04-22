using RayTracingKerr
using Documenter

DocMeta.setdocmeta!(RayTracingKerr, :DocTestSetup, :(using RayTracingKerr); recursive=true)

makedocs(;
    modules=[RayTracingKerr],
    authors="Giorgos Vretinaris",
    repo="https://github.com/gvretina/RayTracingKerr.jl/blob/{commit}{path}#{line}",
    sitename="RayTracingKerr.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gvretina.github.io/RayTracingKerr.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gvretina/RayTracingKerr.jl",
    devbranch="main",
)
