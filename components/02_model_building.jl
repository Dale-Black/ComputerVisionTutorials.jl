### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> title = "Models"
#> description = "Exploration of layer architecture and activation functions."

using Markdown
using InteractiveUtils

# ╔═╡ 403888e0-01ef-4f4c-9bd5-406642bee99a
# ╠═╡ show_logs = false
using Pkg; Pkg.activate(".."); Pkg.instantiate()

# ╔═╡ ffd31741-1b93-42a0-9df5-90cfc571b890
using PlutoUI: TableOfContents

# ╔═╡ 3df1517e-887f-4622-be4b-1a6e7bf93c6b
using Lux

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

# ╔═╡ 97ed35c2-f1e0-4711-b032-7b4cd25b5242
md"""
### 2D
"""

# ╔═╡ 3569b9f3-8efb-4298-ba2f-0c3faf139d59
function conv_layer(
    kernel::Tuple{Int, Int}, in_channels, out_channels;
    pad=2, stride=1, activation=relu)

    return Chain(
        Conv(kernel, in_channels => out_channels, pad=pad, stride=stride),
        BatchNorm(out_channels),
        WrappedFunction(activation)
    )
end

# ╔═╡ f9fe8744-c504-4bac-aaca-e97859045544
md"""
### 3D
"""

# ╔═╡ a05ac626-60d9-4e7f-bcf8-eb9f04030a24
function conv_layer(
    kernel::Tuple{Int, Int, Int}, in_channels, out_channels;
    pad=2, stride=1, activation=relu)

    return Chain(
        Conv(kernel, in_channels => out_channels, pad=pad, stride=stride),
        BatchNorm(out_channels),
        WrappedFunction(activation)
    )
end

# ╔═╡ 66b557b5-905c-4cae-b2b1-fd5180702202
function contract_block(
    kernel::Tuple{Int, Int}, in_channels, mid_channels, out_channels;
    stride=2, activation=relu)

    return Chain(
        conv_layer(kernel, in_channels, mid_channels),
        conv_layer(kernel, mid_channels, out_channels),
        Chain(
            Conv((2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ 2b7cdc35-7fd1-44f6-8bb9-7aaf122fe5a2
function expand_block(
    kernel::Tuple{Int, Int}, in_channels, mid_channels, out_channels;
    stride=2, activation=relu)

    return Chain(
        conv_layer(kernel, in_channels, mid_channels),
        conv_layer(kernel, mid_channels, out_channels),
        Chain(
            ConvTranspose((2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ 9434cfc0-50b7-49c1-9140-a3966d579d41
function contract_block(
    kernel::Tuple{Int, Int, Int}, in_channels, mid_channels, out_channels;
    stride=2, activation=relu)

    return Chain(
        conv_layer(kernel, in_channels, mid_channels),
        conv_layer(kernel, mid_channels, out_channels),
        Chain(
            Conv((2, 2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ 21f36594-872b-43e6-bb76-0ffde1b06782
function expand_block(
    kernel::Tuple{Int, Int, Int}, in_channels, mid_channels, out_channels;
    stride=2, activation=relu)

    return Chain(
        conv_layer(kernel, in_channels, mid_channels),
        conv_layer(kernel, mid_channels, out_channels),
        Chain(
            ConvTranspose((2, 2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ ae1e6464-a0c1-433d-a558-4e550041998f
function Unet(
	kernel::Tuple{Int, Int};
	in_channels::Int = 1, out_channels::Int = in_channels)
	
    return Chain(
        # Initial Convolution Layer
        conv_layer(kernel, in_channels, 4),

        # Contracting Path
        contract_block(kernel, 4, 8, 8),
        contract_block(kernel, 8, 16, 16),
        contract_block(kernel, 16, 32, 32),
        contract_block(kernel, 32, 64, 64),

        # Bottleneck Layer
        conv_layer(kernel, 64, 128),

        # Expanding Path
        expand_block(kernel, 128, 64, 64),
        expand_block(kernel, 64, 32, 32),
        expand_block(kernel, 32, 16, 16),
        expand_block(kernel, 16, 8, 8),

        # Final Convolution Layer
        Conv((1, 1), 8 => out_channels)
    )
end

# ╔═╡ aa4c71df-be71-43f8-bbd7-19737639017d
function Unet(
	kernel::Tuple{Int, Int, Int} = (5, 5, 5);
	in_channels::Int = 1, out_channels::Int = in_channels)
	
    return Chain(
        # Initial Convolution Layer
        conv_layer(kernel, in_channels, 4),

        # Contracting Path
        contract_block(kernel, 4, 8, 8),
        contract_block(kernel, 8, 16, 16),
        contract_block(kernel, 16, 32, 32),
        contract_block(kernel, 32, 64, 64),

        # Bottleneck Layer
        conv_layer(kernel, 64, 128),

        # Expanding Path
        expand_block(kernel, 128, 64, 64),
        expand_block(kernel, 64, 32, 32),
        expand_block(kernel, 32, 16, 16),
        expand_block(kernel, 16, 8, 8),

        # Final Convolution Layer
        Conv((1, 1, 1), 8 => out_channels)
    )
end

# ╔═╡ 4b4f0fde-d267-42ed-8573-d9a81fabb956
Unet((5, 5); in_channels = 1, out_channels = 2)

# ╔═╡ a610688a-71cd-4b8d-8d18-e2866b49b308
Unet((5, 5, 5); in_channels = 1, out_channels = 2)

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
# ╠═3df1517e-887f-4622-be4b-1a6e7bf93c6b
# ╠═0f145b6f-a39e-4771-9181-66464db73f13
# ╟─f2ce7a7c-d0da-43df-b9d0-847aab493ba5
# ╟─e8fbd2ef-e799-4f42-b0df-e68664c43dae
# ╟─ed3b5ebe-d8df-4689-a15a-208caba7701d
# ╟─97ed35c2-f1e0-4711-b032-7b4cd25b5242
# ╠═3569b9f3-8efb-4298-ba2f-0c3faf139d59
# ╠═66b557b5-905c-4cae-b2b1-fd5180702202
# ╠═2b7cdc35-7fd1-44f6-8bb9-7aaf122fe5a2
# ╠═ae1e6464-a0c1-433d-a558-4e550041998f
# ╠═4b4f0fde-d267-42ed-8573-d9a81fabb956
# ╟─f9fe8744-c504-4bac-aaca-e97859045544
# ╠═a05ac626-60d9-4e7f-bcf8-eb9f04030a24
# ╠═9434cfc0-50b7-49c1-9140-a3966d579d41
# ╠═21f36594-872b-43e6-bb76-0ffde1b06782
# ╠═aa4c71df-be71-43f8-bbd7-19737639017d
# ╠═a610688a-71cd-4b8d-8d18-e2866b49b308
# ╟─47f2b0c6-5307-4c66-9620-6e2fad9a53d6
# ╟─0e156ddc-51fb-442c-85e8-eb92187167a7
# ╟─dfb3a032-6a0c-420c-b99e-84a2e61f1f68
