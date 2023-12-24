### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 5658ac7d-2540-4aa6-b3ee-8539ad2cc2dd
using PlutoUI: TableOfContents

# ╔═╡ af20f058-a10a-11ee-3434-077e84c431ac
using MLUtils: DataLoader, splitobs, mapobs

# ╔═╡ fb3d20de-761e-4d40-a8a8-48d76571e67d
using NIfTI: niread

# ╔═╡ d95ace52-2090-4041-afb8-e5a656b2e153
using Glob: glob

# ╔═╡ e129f6e6-4626-4177-8e6a-9807cde05b17
using Random: default_rng, seed!

# ╔═╡ fc87c3b5-db31-4e6f-b8b5-94c64ec3e925
using Lux

# ╔═╡ 6e55628e-b98d-4af6-b255-f2022a24cb89
using DistanceTransforms: transform, boolean_indicator

# ╔═╡ fbd6fcf6-ee02-40b1-8f1d-eac8c47ad485
using ComputerVisionMetrics: hausdorff_metric, dice_metric

# ╔═╡ 7103b96c-494b-4a79-bec8-a9b65a552f9a
using Losers: hausdorff_loss, dice_loss

# ╔═╡ 12d1b039-11a8-4a43-b96e-8ce41a7735f0
using ChainRulesCore: ignore_derivatives

# ╔═╡ cec9523e-f411-40d0-a5f2-ff56571db6f3
# ╠═╡ show_logs = false
using LuxCUDA

# ╔═╡ f0300914-bf8f-4037-a9d4-82fce5dd445c
TableOfContents()

# ╔═╡ 2bd45386-0384-48ed-b074-237b5c07e2c0
begin
	struct ImageCASDataset
		image_paths::Vector{String}
		label_paths::Vector{String}
	end
	
	function ImageCASDataset(root_dir::String)
		image_paths = glob("*.nii*", joinpath(root_dir, "imagesTr"))
		label_paths = glob("*.nii*", joinpath(root_dir, "labelsTr"))
		return ImageCASDataset(image_paths, label_paths)
	end
	
	Base.length(d::ImageCASDataset) = length(d.image_paths)
	
	function Base.getindex(d::ImageCASDataset, i::Int)
	    image = niread(d.image_paths[i]).raw
	    label = niread(d.label_paths[i]).raw
	    return (image, label)
	end
	
	function Base.getindex(d::ImageCASDataset, idxs::AbstractVector{Int})
	    images = Vector{Array{Float32, 3}}(undef, length(idxs))
	    labels = Vector{Array{UInt8, 3}}(undef, length(idxs))
	    for (index, i) in enumerate(idxs)
	        images[index] = niread(d.image_paths[i]).raw
	        labels[index]  = niread(d.label_paths[i]).raw
	    end
	    return (images, labels)
	end

end

# ╔═╡ 8ca62501-62c4-4b43-bc23-c9d1e0e9f14c
md"""
# Data Preparation
"""

# ╔═╡ 84dccd96-b2cc-4d19-8e38-fd0a0a2113f0
md"""
## Dataset
"""

# ╔═╡ e0c124fd-dce9-402f-8cbf-fbe0c1232139
data_dir = "/Users/daleblack/Library/CloudStorage/GoogleDrive-dalejamesblack@gmail.com/My Drive/Datasets/Task02_Heart"

# ╔═╡ 9f9d596c-0c17-4cde-8ee2-8d42eb579517
data = ImageCASDataset(data_dir)

# ╔═╡ 762628a9-7884-42ce-8d6b-bddaf2fe69d1
md"""
## Preprocessing
"""

# ╔═╡ 5fc36915-5295-4fe4-bf22-54f57132023e
function center_crop(volume::Array{T, 3}, target_size::Tuple{Int, Int, Int}) where {T}
    center = div.(size(volume), 2)

    start_idx = max.(1, center .- div.(target_size, 2))
    end_idx = start_idx .+ target_size .- 1

	cropped_volume = volume[start_idx[1]:end_idx[1], start_idx[2]:end_idx[2], start_idx[3]:end_idx[3]]
    return cropped_volume
end

# ╔═╡ 0f7ed804-4bca-41b2-be97-7529b1013b87
function one_hot_encode(label::Array{UInt8, 3}, num_classes::Int)
	one_hot = zeros(UInt8, size(label, 1), size(label, 2), size(label, 3), num_classes)
	
    for k in 1:num_classes
        one_hot[:, :, :, k] = label .== k-1
    end
	
    return one_hot
end

# ╔═╡ 98918966-d3c6-4d2e-b065-f1f3c104d520
function preprocess_image_label_pair(pair, target_size)
    cropped_images = [center_crop(img, target_size) for img in pair[1]]
	cropped_labels = [one_hot_encode(center_crop(lbl, target_size), 2) for lbl in pair[2]]
	
	processed_images = [reshape(img, size(img, 1), size(img, 2), size(img, 3), 1) for img in cropped_images]
	
    return (processed_images, cropped_labels)
end


# ╔═╡ dc8a2d43-90b1-487c-8b4d-05a53ff9f27c
begin
	target_size = (32, 32, 32)
	transformed_data = mapobs(
		x -> preprocess_image_label_pair(x, target_size),
		data
	)
end

# ╔═╡ 95d3943f-da11-4951-b7a9-72c3a7bef31e
md"""
## Dataloaders
"""

# ╔═╡ 47fdf118-84ff-4050-b7ed-21706ffb811a
train_indices, val_indices = splitobs(1:length(data); at = 0.75)

# ╔═╡ e13cd507-a7ac-401d-a1da-1beaa1b3b296
bs = 4

# ╔═╡ 698ba455-3aa2-438f-baa9-40908a7a9686
begin
	train_loader = DataLoader(transformed_data[train_indices]; batchsize = bs, collate = true)
	val_loader = DataLoader(transformed_data[val_indices]; batchsize = bs, collate = true)
end

# ╔═╡ f901815f-885e-4fcb-b68a-2537ed2d8883
md"""
# Model
"""

# ╔═╡ 1e77db93-4681-4885-b13c-1d5b35b8cb3b
md"""
## Randomness
"""

# ╔═╡ 608368e5-548a-4432-ac87-ed91977eeb05
begin
    rng = default_rng()
    seed!(rng, 0)
end

# ╔═╡ 84fb5a87-5643-49d4-9fd5-84864f5c669d
md"""
## Helper functions
"""

# ╔═╡ 943a81cf-6582-4625-b260-55f584b0bd90
function create_unet_layers(
    kernel_size, de_kernel_size, channel_list;
    downsample = true)

    padding = (kernel_size - 1) ÷ 2

	conv1 = Conv((kernel_size, kernel_size, kernel_size), channel_list[1] => channel_list[2], stride=1, pad=padding)
	conv2 = Conv((kernel_size, kernel_size, kernel_size), channel_list[2] => channel_list[3], stride=1, pad=padding)

    relu1 = relu
    relu2 = relu
    bn1 = BatchNorm(channel_list[2])
    bn2 = BatchNorm(channel_list[3])

	bridge_conv = Conv((kernel_size, kernel_size, kernel_size), channel_list[1] => channel_list[3], stride=1, pad=padding)

    if downsample
        sample = Chain(
			Conv((de_kernel_size, de_kernel_size, de_kernel_size), channel_list[3] => channel_list[3], stride=2, pad=(de_kernel_size - 1) ÷ 2, dilation=1),
            BatchNorm(channel_list[3]),
            relu
        )
    else
        sample = Chain(
			ConvTranspose((de_kernel_size, de_kernel_size, de_kernel_size), channel_list[3] => channel_list[3], stride=2, pad=(de_kernel_size - 1) ÷ 2),
            BatchNorm(channel_list[3]),
            relu
        )
    end

    return (conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample)
end

# ╔═╡ ba5f7927-be61-4578-a048-25961b7ef4f3
md"""
## Unet module
"""

# ╔═╡ f345264d-ebb6-41a9-95e6-4e5d684bb088
begin
    struct UNetModule <: Lux.AbstractExplicitContainerLayer{
        (:conv1, :conv2, :bn1, :bn2, :bridge_conv, :sample)
    }
        conv1::Conv
        conv2::Conv
        relu1::Function
        relu2::Function
        bn1::BatchNorm
        bn2::BatchNorm
        bridge_conv::Conv
        sample::Chain
    end

    function UNetModule(
        kernel_size, de_kernel_size, channel_list;
        downsample = true
    )


		conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample = create_unet_layers(
            kernel_size, de_kernel_size, channel_list;
            downsample = downsample
        )

        UNetModule(conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample)
    end

    function (m::UNetModule)(x, ps, st::NamedTuple)
        res, st_bridge_conv = m.bridge_conv(x, ps.bridge_conv, st.bridge_conv)
        x, st_conv1 = m.conv1(x, ps.conv1, st.conv1)
        x, st_bn1 = m.bn1(x, ps.bn1, st.bn1)
        x = relu(x)

        x, st_conv2 = m.conv2(x, ps.conv2, st.conv2)
        x, st_bn2 = m.bn2(x, ps.bn2, st.bn2)
        x = relu(x)

        x = x .+ res

        next_layer, st_sample = m.sample(x, ps.sample, st.sample)

		st = (conv1=st_conv1, conv2=st_conv2, bn1=st_bn1, bn2=st_bn2, bridge_conv=st_bridge_conv, sample=st_sample)
        return next_layer, x, st
    end
end

# ╔═╡ 421f519d-2ff1-4648-931a-1ad961abf4c0
md"""
## Deconv module
"""

# ╔═╡ d43f4c7c-6bd8-4faa-a44c-a79a1e9b9bfb
begin
    struct DeConvModule <: Lux.AbstractExplicitContainerLayer{
        (:conv1, :conv2, :bn1, :bn2, :bridge_conv, :sample)
    }
        conv1::Conv
        conv2::Conv
        relu1::Function
        relu2::Function
        bn1::BatchNorm
        bn2::BatchNorm
        bridge_conv::Conv
        sample::Chain
    end

    function DeConvModule(
        kernel_size, de_kernel_size, channel_list;
        downsample = false)

		conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample = create_unet_layers(
            kernel_size, de_kernel_size, channel_list;
            downsample = downsample
        )

        DeConvModule(conv1, conv2, relu1, relu2, bn1, bn2, bridge_conv, sample)
    end

    function (m::DeConvModule)(x, ps, st::NamedTuple)
        x, x1 = x[1], x[2]
        x = cat(x, x1; dims=4)

        res, st_bridge_conv = m.bridge_conv(x, ps.bridge_conv, st.bridge_conv)

        x, st_conv1 = m.conv1(x, ps.conv1, st.conv1)
        x, st_bn1 = m.bn1(x, ps.bn1, st.bn1)
        x = relu(x)

        x, st_conv2 = m.conv2(x, ps.conv2, st.conv2)
        x, st_bn2 = m.bn2(x, ps.bn2, st.bn2)
        x = relu(x)

        x = x .+ res

        next_layer, st_sample = m.sample(x, ps.sample, st.sample)

		st = (conv1=st_conv1, conv2=st_conv2, bn1=st_bn1, bn2=st_bn2, bridge_conv=st_bridge_conv, sample=st_sample)
        return next_layer, st
    end
end

# ╔═╡ d4f4e929-e295-4198-8ca7-d162e5e57092
md"""
## Model
"""

# ╔═╡ 0a4af070-7672-4a75-b3d5-0aa3257a66dc
begin
    struct FCN <: Lux.AbstractExplicitContainerLayer{
        (:conv1, :conv2, :conv3, :conv4, :conv5, :de_conv1, :de_conv2, :de_conv3, :de_conv4, :last_conv)
    }
        conv1::Chain
        conv2::Chain
        conv3::UNetModule
        conv4::UNetModule
        conv5::UNetModule
        de_conv1::UNetModule
        de_conv2::DeConvModule
        de_conv3::DeConvModule
        de_conv4::DeConvModule
        last_conv::Conv
    end

    function FCN(channel)
        conv1 = Chain(
            Conv((5, 5, 5), 1 => channel, stride=1, pad=2),
            BatchNorm(channel),
            relu
        )
        conv2 = Chain(
            Conv((2, 2, 2), channel => 2 * channel, stride=2, pad=0),
            BatchNorm(2 * channel),
            relu
        )
        conv3 = UNetModule(5, 2, [2 * channel, 2 * channel, 4 * channel])
        conv4 = UNetModule(5, 2, [4 * channel, 4 * channel, 8 * channel])
        conv5 = UNetModule(5, 2, [8 * channel, 8 * channel, 16 * channel])

        de_conv1 = UNetModule(
            5, 2, [16 * channel, 32 * channel, 16 * channel];
            downsample = false
        )
        de_conv2 = DeConvModule(
            5, 2, [32 * channel, 8 * channel, 8 * channel];
            downsample = false
        )
        de_conv3 = DeConvModule(
            5, 2, [16 * channel, 4 * channel, 4 * channel];
            downsample = false
        )
        de_conv4 = DeConvModule(
            5, 2, [8 * channel, 2 * channel, channel];
            downsample = false
        )

        last_conv = Conv((1, 1, 1), 2 * channel => 2, stride=1, pad=0)

		FCN(conv1, conv2, conv3, conv4, conv5, de_conv1, de_conv2, de_conv3, de_conv4, last_conv)
    end

    function (m::FCN)(x, ps, st::NamedTuple)
        # Convolutional layers
        x, st_conv1 = m.conv1(x, ps.conv1, st.conv1)
        x_1 = x  # Store for skip connection
        x, st_conv2 = m.conv2(x, ps.conv2, st.conv2)

        # Downscaling UNet modules
        x, x_2, st_conv3 = m.conv3(x, ps.conv3, st.conv3)
        x, x_3, st_conv4 = m.conv4(x, ps.conv4, st.conv4)
        x, x_4, st_conv5 = m.conv5(x, ps.conv5, st.conv5)

        # Upscaling DeConv modules
        x, _, st_de_conv1 = m.de_conv1(x, ps.de_conv1, st.de_conv1)
        x, st_de_conv2 = m.de_conv2((x, x_4), ps.de_conv2, st.de_conv2)
        x, st_de_conv3 = m.de_conv3((x, x_3), ps.de_conv3, st.de_conv3)
        x, st_de_conv4 = m.de_conv4((x, x_2), ps.de_conv4, st.de_conv4)

        # Concatenate with first skip connection and apply last convolution
        x = cat(x, x_1; dims=4)
        x, st_last_conv = m.last_conv(x, ps.last_conv, st.last_conv)

        # Merge states
        st = (
		conv1=st_conv1, conv2=st_conv2, conv3=st_conv3, conv4=st_conv4, conv5=st_conv5, de_conv1=st_de_conv1, de_conv2=st_de_conv2, de_conv3=st_de_conv3, de_conv4=st_de_conv4, last_conv=st_last_conv
        )

        return x, st
    end
end

# ╔═╡ 0b49da58-7e64-42ed-88fc-970f631c556e
md"""
# Training Set Up
"""

# ╔═╡ ca3835a7-0bda-4a34-b1ec-ddde815a2cbf
import Zygote

# ╔═╡ b3434912-cc78-425d-812f-9ad59c764420
import Optimisers

# ╔═╡ a78d419d-f13d-43d4-8f07-36834b223843
md"""
## Optimiser
"""

# ╔═╡ b8204887-fa61-44c8-a23a-f8302e7b3cf9
function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    return Optimisers.setup(opt, ps)
end

# ╔═╡ f101f985-82d4-4f8d-b4c8-fbddccc4a8db
md"""
## Loss function
"""

# ╔═╡ 4f3c98da-5714-459f-837a-30fdc2041f59
function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)

    # Apply softmax and extract the second channel for binary prediction
    y_pred_softmax = softmax(y_pred, dims=4)
    y_pred_binary = round.(y_pred_softmax[:, :, :, 2, :])

    # Assuming y is already binary and focusing on the second channel
    y_binary = y[:, :, :, 2, :]

    # Compute loss
    loss = 0.0
    for b in axes(y, 5)
        _y_pred = y_pred_binary[:, :, :, b]
        _y = y_binary[:, :, :, b]

		local _y_dtm, _y_pred_dtm
		ignore_derivatives() do
			_y_dtm = transform(boolean_indicator(_y))
			_y_pred_dtm = transform(boolean_indicator(_y_pred))
		end
		
		hd = hausdorff_loss(_y_pred, _y, _y_pred_dtm, _y_dtm)
		dsc = dice_loss(_y_pred, _y)
		loss += hd + dsc
    end
    return loss / size(y, 5), y_pred_binary, st
end

# ╔═╡ 88095e84-b79f-4851-9364-dd3b9a7c7000
md"""
# Train
"""

# ╔═╡ 2051e782-138c-4bd1-8076-34c63bdf3477
dev = gpu_device()

# ╔═╡ d5c4ccb2-128d-44cf-b0e7-62a350dd69c2
model = FCN(4)

# ╔═╡ b670572c-2db6-4e57-a995-a6b72c59961b
begin
	ps, st = Lux.setup(rng, model)
	ps, st = ps |> dev, st |> dev
end

# ╔═╡ a54202f5-d0d8-4792-b54d-f8f5e861537d
function train_model(model, ps, st, train_loader, num_epochs, dev)
    opt_state = create_optimiser(ps)

    for epoch in 1:num_epochs

		# Training Phase
        for (x, y) in train_loader
			x, y = x |> dev, y |> dev
			
            # Forward pass
            y_pred, st = Lux.apply(model, x, ps, st)
            loss, y_pred, st = compute_loss(x, y, model, ps, st)
			@info "Training Loss: $loss"

            # Backward pass
			(loss_grad, st_), back = Zygote.pullback(p -> Lux.apply(model, x, p, st), ps)
            gs = back((one.(loss_grad), nothing))[1]

            # Update parameters
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end

		# Validation Phase
		total_loss = 0.0
		num_batches = 0
	    for (x, y) in val_loader
			x, y = x |> dev, y |> dev
			
	        # Forward Pass
	        y_pred, st = Lux.apply(model, x, ps, st)
	        loss, _, _ = compute_loss(x, y, model, ps, st)
	
	        total_loss += loss
	        num_batches += 1
	    end
		avg_loss = total_loss / num_batches
		@info "Validation Loss: $avg_loss"
    end

    return ps, st
end

# ╔═╡ f3386868-7762-4e00-bdd6-d8bdd180e313
num_epochs = 2

# ╔═╡ 25a0f782-9fe9-4ee4-9b87-f67dea7b0fc7
train_model(model, ps, st, train_loader, num_epochs, dev)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
ComputerVisionMetrics = "56beca70-ca20-45da-83c4-a042539b6c19"
DistanceTransforms = "71182807-4d06-4237-8dd0-bdafe4d097e2"
Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
Losers = "1785af8d-d312-496e-9b53-daf6ddaba92c"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
NIfTI = "a3a9e032-41b5-5fc4-967a-a6b7a19844d3"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
ChainRulesCore = "~1.19.0"
ComputerVisionMetrics = "~0.1.0"
DistanceTransforms = "~0.2.1"
Glob = "~1.3.1"
Losers = "~0.1.0"
Lux = "~0.5.13"
LuxCUDA = "~0.3.1"
MLUtils = "~0.4.3"
NIfTI = "~0.6.0"
Optimisers = "~0.3.1"
PlutoUI = "~0.7.54"
Zygote = "~0.6.68"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0-rc3"
manifest_format = "2.0"
project_hash = "f1fea0820561cffa9fec97f5a58ae875773abdfe"

[[deps.ADTypes]]
git-tree-sha1 = "332e5d7baeff8497b923b730b994fa480601efc7"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.5"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "bbec08a37f8722786d87bedf84eae19c020c4efa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "Statistics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "76582ae19006b1186e87dadd781747f76cead72c"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.1.1"
weakdeps = ["ChainRulesCore", "SpecialFunctions"]

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "1e42ef1bdb45487ff28de16182c0df4920181dc3"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.7.0+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "bcc4a23cbbd99c8535a5318455dcf0f2546ec536"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "9704e50c9158cf8896c2776b8dbc5edd136caf80"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.10.1+0"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "75923dce4275ead3799b238e10178a68c07dbd3b"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "8.9.4+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "0aa0a3dd7b9bacbbadf1932ccbdfa938985c5561"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.58.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "2118cb2765f8197b08e5958cdd17c165427425ee"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.19.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "886826d76ea9e72b35fcd000e535588f7b60f21d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ComputerVisionMetrics]]
deps = ["ImageMorphology", "StatsBase"]
git-tree-sha1 = "523738766a10afbfba028d48981199cade5e531f"
uuid = "56beca70-ca20-45da-83c4-a042539b6c19"
version = "0.1.0"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DistanceTransforms]]
deps = ["GPUArraysCore", "KernelAbstractions", "LoopVectorization"]
git-tree-sha1 = "bdfc622fac116fbebcebbf0c7e00cf59d3ad92f3"
uuid = "71182807-4d06-4237-8dd0-bdafe4d097e2"
version = "0.2.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9a68d75d466ccc1218d0552a8e1631151c569545"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "85d7fb51afb3def5dcb85ad31c3707795c8bccc1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "9.1.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "a846f297ce9d09ccba02ead0cae70690e072a119"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.25.0"

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "8aa91235360659ca7560db43a7d57541120aa31d"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.11"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "fc5d1d3443a124fde6e92d0260cd9e064eba69f8"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.1"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "653e0824fc9ab55b3beec67a6dbbe514a65fb954"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.15"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "0678579657515e88b6632a3a482d39adcbb80445"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.4.1"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "98eaee04d96d973e79c25d49167668c5c8fb50e2"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.27+1"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "62edfee3211981241b57ff1cedf4d74d79519277"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "0f5648fbae0d015e3abe5867bca2b362f67a5894"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.166"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Losers]]
deps = ["Statistics"]
git-tree-sha1 = "7d85766280061a4c145cf4f3478c802423c0540d"
uuid = "1785af8d-d312-496e-9b53-daf6ddaba92c"
version = "0.1.0"

[[deps.Lux]]
deps = ["ADTypes", "Adapt", "ChainRulesCore", "ConcreteStructs", "ConstructionBase", "Functors", "LinearAlgebra", "LuxCore", "LuxDeviceUtils", "LuxLib", "MacroTools", "Markdown", "Optimisers", "PackageExtensionCompat", "Random", "Reexport", "Setfield", "SparseArrays", "Statistics", "TruncatedStacktraces", "WeightInitializers"]
git-tree-sha1 = "08b7e6647e144e31cce846ec4a64cd5d6a77c755"
uuid = "b2108857-7c20-44ae-9111-449ecde12c47"
version = "0.5.13"

    [deps.Lux.extensions]
    LuxChainRulesExt = "ChainRules"
    LuxComponentArraysExt = "ComponentArrays"
    LuxComponentArraysReverseDiffExt = ["ComponentArrays", "ReverseDiff"]
    LuxFluxTransformExt = "Flux"
    LuxLuxAMDGPUExt = "LuxAMDGPU"
    LuxLuxCUDAExt = "LuxCUDA"
    LuxTrackerExt = "Tracker"
    LuxZygoteExt = "Zygote"

    [deps.Lux.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    LuxAMDGPU = "83120cb1-ca15-4f04-bf3b-6967d2e6b60b"
    LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxCUDA]]
deps = ["CUDA", "Reexport", "cuDNN"]
git-tree-sha1 = "69ac44bde022228c11fc8edde18c589f42643e27"
uuid = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
version = "0.3.1"

[[deps.LuxCore]]
deps = ["Functors", "Random", "Setfield"]
git-tree-sha1 = "6feb02e23f6d70f407af97ca270b3a337af5ae0f"
uuid = "bb33d45b-7691-41d6-9220-0943567d0623"
version = "0.1.6"

[[deps.LuxDeviceUtils]]
deps = ["Adapt", "ChainRulesCore", "Functors", "LuxCore", "Preferences", "Random", "SparseArrays"]
git-tree-sha1 = "cffc2222bf9dc01730f6fc904266f8056abb8e83"
uuid = "34f89e08-e1d5-43b4-8944-0b49ac560553"
version = "0.1.11"

    [deps.LuxDeviceUtils.extensions]
    LuxDeviceUtilsFillArraysExt = "FillArrays"
    LuxDeviceUtilsLuxAMDGPUExt = "LuxAMDGPU"
    LuxDeviceUtilsLuxCUDAExt = "LuxCUDA"
    LuxDeviceUtilsMetalExt = "Metal"
    LuxDeviceUtilsZygoteExt = "Zygote"

    [deps.LuxDeviceUtils.weakdeps]
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    LuxAMDGPU = "83120cb1-ca15-4f04-bf3b-6967d2e6b60b"
    LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxLib]]
deps = ["ChainRulesCore", "KernelAbstractions", "Markdown", "NNlib", "PackageExtensionCompat", "PrecompileTools", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7dc09243c8b6b4b71d1bac27c8e4593b42379d2e"
uuid = "82251201-b29d-42c6-8e01-566dec8acb11"
version = "0.3.8"

    [deps.LuxLib.extensions]
    LuxLibForwardDiffExt = "ForwardDiff"
    LuxLibLuxCUDAExt = "LuxCUDA"
    LuxLibLuxCUDATrackerExt = ["LuxCUDA", "Tracker"]
    LuxLibReverseDiffExt = "ReverseDiff"
    LuxLibTrackerExt = "Tracker"

    [deps.LuxLib.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "3504cdb8c2bc05bde4d4b09a81b01df88fcbbba0"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.3"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "b211c553c199c111d998ecdaf7623d1b89b69f93"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.12"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NIfTI]]
deps = ["Base64", "CodecZlib", "MappedArrays", "Mmap", "TranscodingStreams"]
git-tree-sha1 = "21e5b879564607ea98fb680c98a1b7838b7d7f1c"
uuid = "a3a9e032-41b5-5fc4-967a-a6b7a19844d3"
version = "0.6.0"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "7c221293228506db2fe883251407581e0846688e"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.9"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "8bc9ce4233be3c63f8dcd78ccaf1b63a9c0baa34"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "0.3.3"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ce3269ed42816bf18d500c9f63418d4b0d9f5a3b"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.1.0+2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "6a731f2b5c03157418a20c12195eb4b74c8f8621"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.13.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "34205b1204cc83c43cd9cfe53ffbd3b310f6e8c5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.1"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PartialFunctions]]
deps = ["MacroTools"]
git-tree-sha1 = "47b49a4dbc23b76682205c646252c0f9e1eb75af"
uuid = "570af359-4316-4cb7-8c74-252c00c2016b"
version = "1.2.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "c860e84651f58ce240dd79e5d9e055d55234c35a"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.2"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "3aac6d68c5e57449f5b9b865c9ba50ac2970c4cf"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.42"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "5d66818a39bb04bf328e92bc933ec5b4ee88e436"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "fba11dbe2562eecdfcac49a05246af09ee64d055"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.8.1"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "e579d3c991938fecbb225699e8f611fa3fbf2141"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.79"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "7209df901e6ed7489fe9b7aa3e46fb788e15db85"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.65"

[[deps.WeightInitializers]]
deps = ["PackageExtensionCompat", "PartialFunctions", "Random", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "f5c5118f3cd9a2ee5992d43d0a296caa671edea9"
uuid = "d49dbf32-c5c2-4618-8acc-27bb2598ef2d"
version = "0.1.3"
weakdeps = ["CUDA"]

    [deps.WeightInitializers.extensions]
    WeightInitializersCUDAExt = "CUDA"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "30c1b8bfc2b3c7c5d8bba7cd32e8b6d5f968e7c3"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.68"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "9d749cd449fb448aeca4feee9a2f4186dbb5d184"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.4"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDA_Runtime_Discovery", "CUDNN_jll"]
git-tree-sha1 = "c092c26591a851083ed3358890d0d916c58dde62"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.2.1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═5658ac7d-2540-4aa6-b3ee-8539ad2cc2dd
# ╠═f0300914-bf8f-4037-a9d4-82fce5dd445c
# ╟─8ca62501-62c4-4b43-bc23-c9d1e0e9f14c
# ╠═af20f058-a10a-11ee-3434-077e84c431ac
# ╠═fb3d20de-761e-4d40-a8a8-48d76571e67d
# ╠═d95ace52-2090-4041-afb8-e5a656b2e153
# ╟─84dccd96-b2cc-4d19-8e38-fd0a0a2113f0
# ╠═2bd45386-0384-48ed-b074-237b5c07e2c0
# ╠═e0c124fd-dce9-402f-8cbf-fbe0c1232139
# ╠═9f9d596c-0c17-4cde-8ee2-8d42eb579517
# ╟─762628a9-7884-42ce-8d6b-bddaf2fe69d1
# ╠═5fc36915-5295-4fe4-bf22-54f57132023e
# ╠═0f7ed804-4bca-41b2-be97-7529b1013b87
# ╠═98918966-d3c6-4d2e-b065-f1f3c104d520
# ╠═dc8a2d43-90b1-487c-8b4d-05a53ff9f27c
# ╟─95d3943f-da11-4951-b7a9-72c3a7bef31e
# ╠═47fdf118-84ff-4050-b7ed-21706ffb811a
# ╠═e13cd507-a7ac-401d-a1da-1beaa1b3b296
# ╠═698ba455-3aa2-438f-baa9-40908a7a9686
# ╟─f901815f-885e-4fcb-b68a-2537ed2d8883
# ╠═e129f6e6-4626-4177-8e6a-9807cde05b17
# ╠═fc87c3b5-db31-4e6f-b8b5-94c64ec3e925
# ╟─1e77db93-4681-4885-b13c-1d5b35b8cb3b
# ╠═608368e5-548a-4432-ac87-ed91977eeb05
# ╟─84fb5a87-5643-49d4-9fd5-84864f5c669d
# ╠═943a81cf-6582-4625-b260-55f584b0bd90
# ╟─ba5f7927-be61-4578-a048-25961b7ef4f3
# ╠═f345264d-ebb6-41a9-95e6-4e5d684bb088
# ╟─421f519d-2ff1-4648-931a-1ad961abf4c0
# ╠═d43f4c7c-6bd8-4faa-a44c-a79a1e9b9bfb
# ╟─d4f4e929-e295-4198-8ca7-d162e5e57092
# ╠═0a4af070-7672-4a75-b3d5-0aa3257a66dc
# ╟─0b49da58-7e64-42ed-88fc-970f631c556e
# ╠═6e55628e-b98d-4af6-b255-f2022a24cb89
# ╠═fbd6fcf6-ee02-40b1-8f1d-eac8c47ad485
# ╠═7103b96c-494b-4a79-bec8-a9b65a552f9a
# ╠═12d1b039-11a8-4a43-b96e-8ce41a7735f0
# ╠═ca3835a7-0bda-4a34-b1ec-ddde815a2cbf
# ╠═b3434912-cc78-425d-812f-9ad59c764420
# ╟─a78d419d-f13d-43d4-8f07-36834b223843
# ╠═b8204887-fa61-44c8-a23a-f8302e7b3cf9
# ╟─f101f985-82d4-4f8d-b4c8-fbddccc4a8db
# ╠═4f3c98da-5714-459f-837a-30fdc2041f59
# ╟─88095e84-b79f-4851-9364-dd3b9a7c7000
# ╠═cec9523e-f411-40d0-a5f2-ff56571db6f3
# ╠═2051e782-138c-4bd1-8076-34c63bdf3477
# ╠═d5c4ccb2-128d-44cf-b0e7-62a350dd69c2
# ╠═b670572c-2db6-4e57-a995-a6b72c59961b
# ╠═a54202f5-d0d8-4792-b54d-f8f5e861537d
# ╠═f3386868-7762-4e00-bdd6-d8bdd180e313
# ╠═25a0f782-9fe9-4ee4-9b87-f67dea7b0fc7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
