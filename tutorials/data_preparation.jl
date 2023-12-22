### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 7cf70220-ce1b-4a51-9354-c9c8585ba1c3
# ╠═╡ show_logs = false
using Pkg; Pkg.activate("..")

# ╔═╡ b4f772c8-9448-424c-a669-c7f8c0cb2133
using Lux

# ╔═╡ 1dfe8f20-58e4-4236-8f7a-e5fa5448b018
using PlutoUI: TableOfContents

# ╔═╡ 1b25f942-0ddf-4242-b479-56d76eff5dbc
using DistanceTransforms: transform, boolean_indicator

# ╔═╡ d78662a3-cac4-49d7-a8e3-a264ebb51cd8
using Losers: dice_loss, hausdorff_loss

# ╔═╡ 4279cd19-eff0-48d5-8742-17c83ae6f43e
using ComputerVisionMetrics: dice_metric, hausdorff_metric

# ╔═╡ 44d1d71c-271a-4e50-ab07-b9778eeb1616
TableOfContents()

# ╔═╡ 1aafbadc-7857-4685-915c-ccc5b2a58551
md"""
# Data Preparation

This tutorial provides a comprehensive outline for how users might utilize packages like MLUtils.jl, NIfTI, etc. for preparing data for deep learning training.
"""

# ╔═╡ 844d42a7-173b-4f4c-9d4f-d2b3f475cc36


# ╔═╡ Cell order:
# ╠═7cf70220-ce1b-4a51-9354-c9c8585ba1c3
# ╠═b4f772c8-9448-424c-a669-c7f8c0cb2133
# ╠═1dfe8f20-58e4-4236-8f7a-e5fa5448b018
# ╠═1b25f942-0ddf-4242-b479-56d76eff5dbc
# ╠═d78662a3-cac4-49d7-a8e3-a264ebb51cd8
# ╠═4279cd19-eff0-48d5-8742-17c83ae6f43e
# ╠═44d1d71c-271a-4e50-ab07-b9778eeb1616
# ╟─1aafbadc-7857-4685-915c-ccc5b2a58551
# ╠═844d42a7-173b-4f4c-9d4f-d2b3f475cc36
