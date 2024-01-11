### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> title = "Data Preparation"
#> description = "Dataset creation, preprocessing, augmentation, data loaders, & more..."

using Markdown
using InteractiveUtils

# ╔═╡ 7cf70220-ce1b-4a51-9354-c9c8585ba1c3
# ╠═╡ show_logs = false
using Pkg; Pkg.activate("."); Pkg.instantiate()

# ╔═╡ 6d8f9217-86c2-46cb-876c-ed76b08b2093
using PlutoUI: TableOfContents

# ╔═╡ 44d1d71c-271a-4e50-ab07-b9778eeb1616
TableOfContents()

# ╔═╡ 1aafbadc-7857-4685-915c-ccc5b2a58551
md"""
# Data Preparation

This tutorial provides a comprehensive outline for how users might utilize packages like MLUtils.jl, NIfTI, etc. for preparing data for deep learning training.
"""

# ╔═╡ 844d42a7-173b-4f4c-9d4f-d2b3f475cc36
md"""
## Dataset Creation
"""

# ╔═╡ 5a3c55b6-f9d6-48d8-9da4-e0de7875b2fa
md"""
## Preprocessing
"""

# ╔═╡ 79dfbb24-0cd1-4354-8498-f9bcb2828091
md"""
## Augmentation
"""

# ╔═╡ 48be6648-3c62-4e7c-8cd7-1788c035a66d
md"""
## Dataloaders
"""

# ╔═╡ Cell order:
# ╠═7cf70220-ce1b-4a51-9354-c9c8585ba1c3
# ╠═6d8f9217-86c2-46cb-876c-ed76b08b2093
# ╠═44d1d71c-271a-4e50-ab07-b9778eeb1616
# ╟─1aafbadc-7857-4685-915c-ccc5b2a58551
# ╟─844d42a7-173b-4f4c-9d4f-d2b3f475cc36
# ╟─5a3c55b6-f9d6-48d8-9da4-e0de7875b2fa
# ╟─79dfbb24-0cd1-4354-8498-f9bcb2828091
# ╟─48be6648-3c62-4e7c-8cd7-1788c035a66d
