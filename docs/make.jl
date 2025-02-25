using FredholmDeterminants
using Documenter

DocMeta.setdocmeta!(FredholmDeterminants, :DocTestSetup, :(using FredholmDeterminants); recursive=true)

makedocs(;
    modules=[FredholmDeterminants],
    authors="Simeon David Schaub <simeon@schaub.rocks> and contributors",
    sitename="FredholmDeterminants.jl",
    format=Documenter.HTML(;
        canonical="https://simeonschaub.github.io/FredholmDeterminants.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/simeonschaub/FredholmDeterminants.jl",
    devbranch="main",
)
