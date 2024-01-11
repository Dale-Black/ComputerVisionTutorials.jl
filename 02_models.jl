### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> title = "Models"
#> description = "Common deep learning models for computer vision"

using Markdown
using InteractiveUtils

# ╔═╡ 403888e0-01ef-4f4c-9bd5-406642bee99a
# ╠═╡ show_logs = false
using Pkg; Pkg.activate("."); Pkg.instantiate()

# ╔═╡ ffd31741-1b93-42a0-9df5-90cfc571b890
using PlutoUI: TableOfContents

# ╔═╡ 0f145b6f-a39e-4771-9181-66464db73f13
TableOfContents()

# ╔═╡ f2ce7a7c-d0da-43df-b9d0-847aab493ba5
md"""
# Models

This tutorial shows the basic steps in building and validating common model types for medical imaging based deep learning
"""

# ╔═╡ e8fbd2ef-e799-4f42-b0df-e68664c43dae
md"""
## FCN
"""

# ╔═╡ ed3b5ebe-d8df-4689-a15a-208caba7701d
md"""
## UNet
"""

# ╔═╡ 47f2b0c6-5307-4c66-9620-6e2fad9a53d6
md"""
## Vision Transformer
"""

# ╔═╡ 0e156ddc-51fb-442c-85e8-eb92187167a7
md"""
## Diffusion
"""

# ╔═╡ dfb3a032-6a0c-420c-b99e-84a2e61f1f68
md"""
## State Space
"""

# ╔═╡ Cell order:
# ╠═403888e0-01ef-4f4c-9bd5-406642bee99a
# ╠═ffd31741-1b93-42a0-9df5-90cfc571b890
# ╠═0f145b6f-a39e-4771-9181-66464db73f13
# ╟─f2ce7a7c-d0da-43df-b9d0-847aab493ba5
# ╟─e8fbd2ef-e799-4f42-b0df-e68664c43dae
# ╟─ed3b5ebe-d8df-4689-a15a-208caba7701d
# ╟─47f2b0c6-5307-4c66-9620-6e2fad9a53d6
# ╟─0e156ddc-51fb-442c-85e8-eb92187167a7
# ╟─dfb3a032-6a0c-420c-b99e-84a2e61f1f68
