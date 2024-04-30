### A Pluto.jl notebook ###
# v0.19.40

#> [frontmatter]
#> title = "Image Segmentation"
#> description = "Guide on 3D heart segmentation in CT images."

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 8d4a6d5a-c437-43bb-a3db-ab961b218c2e
using PlutoUI: TableOfContents, Slider, bind

# ╔═╡ 83b95cee-90ed-4522-b9a8-79c082fce02e
using Random: default_rng, seed!

# ╔═╡ 7353b7ce-8b33-4602-aed7-2aa24864aca5
using HTTP: download

# ╔═╡ de5efc37-db19-440e-9487-9a7bea84996d
using Tar: extract

# ╔═╡ db2ccf3a-437a-4dfa-ad05-2526c0e2bde0
using Glob: glob

# ╔═╡ 562b3772-89cc-4390-87c3-e7260c8aa86b
using NIfTI: niread

# ╔═╡ 3ab44a2a-692f-4603-a5a8-81f1d260c13e
using MLUtils: DataLoader, splitobs, mapobs, getobs

# ╔═╡ da9cada1-7ea0-4b6b-a338-d8e08b668d28
using ImageTransformations: imresize

# ╔═╡ 6f6e49fc-3322-4da7-b6ff-8846260139b2
using ImageFiltering: imfilter, KernelFactors.gaussian

# ╔═╡ 8e2f2c6d-127d-42a6-9906-970c09a22e61
using CairoMakie: Figure, Axis, heatmap!

# ╔═╡ a3f44d7c-efa3-41d0-9509-b099ab7f09d4
using Lux

# ╔═╡ 317c1571-d232-4cab-ac10-9fc3b7ad33b0
# ╠═╡ show_logs = false
using LuxCUDA

# ╔═╡ 12d42392-ad7b-4c5f-baf5-1f2c6052669e
using Optimisers: Adam, setup

# ╔═╡ a6669580-de24-4111-a7cb-26d3e727a12e
using DistanceTransforms: transform, boolean_indicator

# ╔═╡ 70bc36db-9ee3-4e1d-992d-abbf55c52070
using Losers: hausdorff_loss, dice_loss

# ╔═╡ 2f6f0755-d71f-4239-a72b-88a545ba8ca1
using Dates: now

# ╔═╡ 69880e6d-162a-4aae-94eb-103bd35ac3c9
using Zygote: pullback

# ╔═╡ dce913e0-126d-4aa3-933a-4f07eea1b8ae
using Optimisers: update

# ╔═╡ c283f9a3-6a76-4186-859f-21cd9efc131f
using ChainRulesCore: ignore_derivatives

# ╔═╡ dfc9377a-7cc1-43ba-bb43-683d24e67d79
using ComputerVisionMetrics: hausdorff_metric, dice_metric

# ╔═╡ e457a411-2e7b-43b3-a247-23eff94222b0
using DataFrames: DataFrame

# ╔═╡ 1b5ae165-1069-4638-829a-471b907cce86
using CSV: write

# ╔═╡ b04c696b-b404-4976-bfc1-51889ef1d60f
using JLD2: jldsave

# ╔═╡ c4824b83-01aa-411d-b088-1e5320224e3c
using CSV: read

# ╔═╡ 3d4f7938-f7f6-47f1-ad1d-c56a7d7a987f
using CairoMakie: scatterlines!, lines!, axislegend, ylims!

# ╔═╡ 00ea61c1-7d20-4c98-892e-dcdec3b0b43f
using FileIO: load

# ╔═╡ 51e6af07-a448-400c-9073-1a7b2c0d69c8
# using Pkg; Pkg.activate(; temp = true)

# ╔═╡ c8d6553a-90df-4aeb-aa6d-a213e16fab48
TableOfContents()

# ╔═╡ af50e5f3-1a1c-47e5-a461-ffbee0329309
begin
    rng = default_rng()
    seed!(rng, 0)
end

# ╔═╡ cdfd2412-897d-4642-bb69-f8031c418446
function download_dataset(heart_url, target_directory)
    if isempty(readdir(target_directory))
        local_tar_file = joinpath(target_directory, "heart_dataset.tar")
		download(heart_url, "heart_dataset.tar")
		extract("heart_dataset.tar", target_directory)
		data_dir = joinpath(target_directory, readdir(target_directory)...)
        return data_dir
    else
        @warn "Target directory is not empty. Aborting download and extraction."
        return joinpath(target_directory, readdir(target_directory)...)
    end
end

# ╔═╡ b1516500-ad83-41d2-8a1d-093cd0d948e3
heart_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar"

# ╔═╡ 3e896957-61d8-4750-89bd-be02383417ec
# target_directory = mktempdir()

# ╔═╡ 4e715848-611a-4125-8ee6-ac5b4d3e4147
target_directory = "/dfs7/symolloi-lab/msd_heart_dir"

# ╔═╡ 99211382-7de9-4e97-872f-d0c01b8f8307
# ╠═╡ show_logs = false
data_dir = download_dataset(heart_url, target_directory)

# ╔═╡ 6d34b756-4da8-427c-91f5-dfb022c4e715
begin
	struct HeartSegmentationDataset
		image_paths::Vector{String}
		label_paths::Vector{String}
	end
	
	function HeartSegmentationDataset(root_dir::String; is_test::Bool = false)
	    if is_test
	        image_paths = glob("*.nii*", joinpath(root_dir, "imagesTs"))
	        return HeartSegmentationDataset(image_paths, String[])
	    else
	        image_paths = glob("*.nii*", joinpath(root_dir, "imagesTr"))
	        label_paths = glob("*.nii*", joinpath(root_dir, "labelsTr"))
	        return HeartSegmentationDataset(image_paths, label_paths)
	    end
	end
	
	Base.length(d::HeartSegmentationDataset) = length(d.image_paths)
	
	function Base.getindex(d::HeartSegmentationDataset, i::Int)
	    image = niread(d.image_paths[i]).raw
	    if !isempty(d.label_paths)
	        label = niread(d.label_paths[i]).raw
	        return (image, label)
	    else
	        return image
	    end
	end
	
	function Base.getindex(d::HeartSegmentationDataset, idxs::AbstractVector{Int})
	    images = Vector{Array{Float32, 3}}(undef, length(idxs))
		labels = isempty(d.label_paths) ? nothing : Vector{Array{UInt8, 3}}(undef, length(idxs))
	
	    for (index, i) in enumerate(idxs)
	        images[index] = niread(d.image_paths[i]).raw
	        if labels !== nothing
	            labels[index] = niread(d.label_paths[i]).raw
	        end
	    end
	
	    if labels !== nothing
	        return (images, labels)
	    else
	        return images
	    end
	end
end

# ╔═╡ 65dac38d-f955-4058-b577-827d7f8b3db4
md"""
# Set Up
"""

# ╔═╡ 7cf78ac3-cedd-479d-bc50-769f7b772060
md"""
## Imports
"""

# ╔═╡ fc917017-4d02-4c2d-84d6-b5497d825fff
md"""
!!! info
	Pluto automatically handles the installation of all of the packages imported throughout the notebook. You might notice that the first time running this notebook takes a while to get started. This is likely because all of these packages are being installed.md

	After the first time, the loading time will be much quicker.
"""

# ╔═╡ af798f6b-7549-4253-b02b-2ed20dc1125b
md"""
## Randomness
"""

# ╔═╡ f0e64ba5-5e11-4ddb-91d3-2a34c60dc6bf
md"""
# 1. Data Preparation
"""

# ╔═╡ ec7734c3-33a5-43c7-82db-2db4dbdc9587
md"""
## Dataset
"""

# ╔═╡ 9577b91b-faa4-4fc5-9ec2-ed8ca94f2afe
data = HeartSegmentationDataset(data_dir)

# ╔═╡ 0e820544-dc33-43fb-85be-f928758b8b67
md"""
## Data Preprocessing
"""

# ╔═╡ cf1b6b00-d55c-4310-b1e6-ca03a009a098
function crop(
    img, label=nothing;
    target_size=nothing, max_crop_percentage=0.15)
	
	current_size = size(img)
	N = length(current_size)
	
	if isnothing(target_size)
		target_size = current_size
	elseif length(target_size) != N
		throw(ArgumentError("The dimensionality of target_size must match the dimensionality of the input image."))
	end
	
	# Calculate the maximum allowable crop size, only for dimensions larger than target size
	max_crop_size = [min(cs, round(Int, ts + (cs - ts) * max_crop_percentage)) for (cs, ts) in zip(current_size, target_size)]
	
	# Center crop for dimensions needing cropping
	start_idx = [
		cs > ts ? max(1, div(cs, 2) - div(ms, 2)) : 1 for (cs, ts, ms) in zip(current_size, target_size, max_crop_size)
	]
	end_idx = start_idx .+ max_crop_size .- 1
	
	# Create a tuple of ranges for indexing
	crop_ranges = tuple([start_idx[i]:end_idx[i] for i in 1:N]...)
	
	cropped_img = img[crop_ranges...]
	cropped_label = isnothing(label) ? nothing : label[crop_ranges...]
	
	return cropped_img, cropped_label
end

# ╔═╡ b4d6f9da-677d-494f-bbc7-811eb65a6bd7
function resize(
	img, label=nothing;
	target_size=(128, 128, 96))
	
    resized_img = imresize(img, target_size)
    resized_label = isnothing(label) ? nothing : imresize(label, target_size)
    return resized_img, resized_label
end

# ╔═╡ 3e1b60c0-ea32-4024-b6f7-44d3257a44ac
function one_hot_encode(
	img, label;
	num_classes=2)

    img_one_hot = reshape(img, size(img)..., 1)

    if isnothing(label)
        return img_one_hot, nothing
    else
        label_one_hot = Float32.(zeros(size(label)..., num_classes))

		if ndims(label) == 2
	        for k in 1:num_classes
	            label_one_hot[:, :, k] = Float32.(label .== (k-1))
	        end
		elseif ndims(label) == 3
			for k in 1:num_classes
	            label_one_hot[:, :, :, k] = Float32.(label .== (k-1))
	        end
		end

        return img_one_hot, label_one_hot
    end
end

# ╔═╡ c5bc3f84-8679-4ff7-863e-67d6125e1a4f
function preprocess_data(
    img, label = nothing;
    target_size = (128, 128, 96))
	
	img, label = crop(img, label; target_size = target_size)
	img, label = resize(img, label; target_size = target_size)
	img, label = one_hot_encode(img, label; num_classes = 2)
    return Float32.(img), Float32.(label)
end

# ╔═╡ ea0fd7c2-7cbe-4e30-905e-457ec81b42c5
target_size = (128, 128, 96)

# ╔═╡ 43cf82c5-f0ef-42dd-ad5c-6265d345da9e
preprocessed_data = mapobs(pair -> preprocess_data(pair...), data)

# ╔═╡ f48d5547-a80c-4709-aa2c-0dd4a5b2d2a7
# image_pre, label_pre = getobs(preprocessed_data, 1);

# ╔═╡ 733e6868-6bd4-4b4a-b1a5-815db1cd8286
preprocessed_train_data, preprocessed_val_data = splitobs(preprocessed_data; at = 0.75)

# ╔═╡ 8d97c2b5-659f-42d8-a86b-00638790b62f
md"""
## Data Augmentation
"""

# ╔═╡ 8e5d073b-98ff-412e-b9fe-70e6e9e912f4
function rand_gaussian_blur(img, label=nothing; p=0.5, k=3, σ=0.3 * ((k - 1) / 2 - 1) + 0.8)
	if rand() < p
		if isa(k, Integer)
			k = fill(k, ndims(img))
		end
		if isa(σ, Real)
			σ = fill(σ, ndims(img))
		end
	
		minimum(k) > 0 || throw(ArgumentError("Kernel size must be positive: $(k)"))
		minimum(σ) > 0 || throw(ArgumentError("σ must be positive: $(σ)"))
	
		kernel = gaussian(σ, k)
		blurred_img = imfilter(img, kernel)
		blurred_label = isnothing(label) ? nothing : round.(Int, imfilter(Float32.(label), kernel))
		return blurred_img, blurred_label
	else
		return img, label
	end
end

# ╔═╡ 5996e996-5d79-48a4-90de-2b07d9b5d59e
function rand_flip_x(img, label=nothing; p=0.5)
    if rand() < p
        flipped_img = reverse(img; dims=2)
        flipped_label = isnothing(label) ? nothing : reverse(label; dims=2)
        return flipped_img, flipped_label
    else
        return img, label
    end
end

# ╔═╡ 28592dec-220c-46c5-9e7f-51774e134ff1
function rand_flip_y(img, label=nothing; p=0.5)
    if rand() < p
        flipped_img = reverse(img; dims=1)
        flipped_label = isnothing(label) ? nothing : reverse(label; dims=1)
        return flipped_img, flipped_label
    else
        return img, label
    end
end

# ╔═╡ f191b8dc-5ba5-431e-a46a-cf5109d6fc7b
function augment_data(
	img, label=nothing;
	blur_prob=0.5,
	flip_x_prob=0.5,
	flip_y_prob=0.5,
	shear_x_prob=0.5,
	shear_y_prob=0.5)
	
    img, label = rand_gaussian_blur(img, label; p=blur_prob)
    img, label = rand_flip_x(img, label; p=flip_x_prob)
    img, label = rand_flip_y(img, label; p=flip_y_prob)
    return Float32.(img), Float32.(label)
end

# ╔═╡ c1d624c8-4e1b-44d3-8f8d-dce740841a20
augmented_train_data = mapobs(pair -> augment_data(pair...), preprocessed_train_data)

# ╔═╡ 274f277b-0bda-47e7-a52a-e75be9538957
# image_aug, label_aug = getobs(augmented_data, 1);

# ╔═╡ 03bab55a-6e5e-4b9f-b56a-7e9f993576eb
md"""
## Dataloaders
"""

# ╔═╡ cf23fca5-78f6-4bc4-9f9b-24c062254a58
bs = 1

# ╔═╡ 2032b7e6-ceb7-4c08-9b0d-bc704f5e4104
begin
	train_loader = DataLoader(augmented_train_data; batchsize = bs, collate = true)
	val_loader = DataLoader(preprocessed_val_data; batchsize = bs, collate = true)
end

# ╔═╡ 2ec43028-c1ab-4df7-9cfe-cc1a4919a7cf
md"""
## Visualize
"""

# ╔═╡ a6316144-c809-4d2a-bda1-d5128dcf89d3
md"""
### Original Data
"""

# ╔═╡ f8fc2cee-c1bd-477d-9595-9427e8764bd6
image_raw, label_raw = getobs(data, 1);

# ╔═╡ 7cb986f8-b338-4046-b569-493e443a8dcb
@bind z1 Slider(axes(image_raw, 3), show_value = true, default = div(size(image_raw, 3), 2))

# ╔═╡ d7e75a72-8281-432c-abab-c254f8c94d3c
let
	f = Figure(size = (700, 500))
	ax = Axis(
		f[1, 1],
		title = "Original Image"
	)
	heatmap!(image_raw[:, :, z1]; colormap = :grays)

	ax = Axis(
		f[1, 2],
		title = "Original Label (Overlayed)"
	)
	heatmap!(image_raw[:, :, z1]; colormap = :grays)
	heatmap!(label_raw[:, :, z1]; colormap = (:jet, 0.4))
	f
end

# ╔═╡ 9dc89870-3d99-472e-8974-712e34a3a789
md"""
### Transformed Data
"""

# ╔═╡ 0f5d7796-2c3d-4b74-86c1-a1d4e3922011
image_tfm, label_tfm = getobs(augmented_train_data, 1);

# ╔═╡ 51e9e7d9-a1d2-4fd1-bdad-52851d9498a6
typeof(image_tfm), typeof(label_tfm)

# ╔═╡ 803d918a-66ce-4ed3-a33f-5dda2dd7288e
unique(label_tfm)

# ╔═╡ 6e2bfcfb-77e3-4532-a14d-10f4b91f2f54
@bind z2 Slider(1:target_size[3], show_value = true, default = div(target_size[3], 2))

# ╔═╡ bae79c05-034a-4c39-801a-01229b618e94
let
	f = Figure(size = (700, 500))
	ax = Axis(
		f[1, 1],
		title = "Transformed Image"
	)
	heatmap!(image_tfm[:, :, z2, 1]; colormap = :grays)

	ax = Axis(
		f[1, 2],
		title = "Transformed Label (Overlayed)"
	)
	heatmap!(image_tfm[:, :, z2, 1]; colormap = :grays)
	heatmap!(label_tfm[:, :, z2, 2]; colormap = (:jet, 0.4))
	f
end

# ╔═╡ 95ad5275-63ca-4f2a-9f3e-6c6a340f5cd4
md"""
# 2. Model Building
"""

# ╔═╡ 773aace6-14ad-46f6-a1a6-692247231e90
md"""
## Helper Blocks
"""

# ╔═╡ 1588d84a-c5f7-4be6-9295-c3594d77b08f
function conv_layer(
	k, in_channels, out_channels;
	pad=2, stride=1, activation=relu)
	
    return Chain(
        Conv((k, k, k), in_channels => out_channels, pad=pad, stride=stride),
        BatchNorm(out_channels),
        WrappedFunction(activation)
    )
end

# ╔═╡ f9b0aa7f-d660-4d6f-bd5d-721e5c809b13
function contract_block(
	in_channels, mid_channels, out_channels;
	k=5, stride=2, activation=relu)
	
    return Chain(
        conv_layer(k, in_channels, mid_channels),
        conv_layer(k, mid_channels, out_channels),
        Chain(
            Conv((2, 2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ e682f461-43d7-492a-85a9-2d46e829a125
function expand_block(
	in_channels, mid_channels, out_channels;
	k=5, stride=2, activation=relu)
	
    return Chain(
        conv_layer(k, in_channels, mid_channels),
        conv_layer(k, mid_channels, out_channels),
        Chain(
            ConvTranspose((2, 2, 2), out_channels => out_channels, stride=stride),
            BatchNorm(out_channels),
            WrappedFunction(activation)
        )
    )
end

# ╔═╡ 36ad66d6-c484-4073-bf01-1f7ec7012373
md"""
## Unet
"""

# ╔═╡ f55e3c0f-6abe-423c-8319-96146f30eebd
function Unet(in_channels::Int = 1, out_channels::Int = in_channels)
    return Chain(
        # Initial Convolution Layer
        conv_layer(5, in_channels, 4),

        # Contracting Path
        contract_block(4, 8, 8),
        contract_block(8, 16, 16),
        contract_block(16, 32, 32),
        contract_block(32, 64, 64),

        # Bottleneck Layer
        conv_layer(5, 64, 128),

        # Expanding Path
        expand_block(128, 64, 64),
        expand_block(64, 32, 32),
        expand_block(32, 16, 16),
        expand_block(16, 8, 8),

        # Final Convolution Layer
        Conv((1, 1, 1), 8 => out_channels)
    )
end

# ╔═╡ bbdaf5c5-9faa-4b61-afab-c0242b8ca034
model = Unet(1, 2)

# ╔═╡ df2dd9a7-045c-44a5-a62c-8d9f2541dc14
md"""
# 3. Training & Validation
"""

# ╔═╡ 7cde37c8-4c59-4583-8995-2b01eda95cb3
md"""
## Optimiser
"""

# ╔═╡ 0390bcf5-4cd6-49ba-860a-6f94f8ba6ded
function create_optimiser(ps)
	opt = Adam(0.001f0)
    return setup(opt, ps)
end

# ╔═╡ a25bdfe6-b24d-446b-926f-6e0727d647a2
md"""
## Loss function
"""

# ╔═╡ 08f2911c-90e7-418e-b9f2-a0722a857bf1
function compute_loss(x, y, model, ps, st)
    # Get model predictions
	y_pred, st = Lux.apply(model, x, ps, st)

    # Apply sigmoid activation
    y_pred_sigmoid = sigmoid.(y_pred)

    # Compute loss
    loss = 0.0
    for b in axes(y, 5)  # Iterate over the batch dimension
        _y_pred = y_pred_sigmoid[:, :, :, 2, b]
        _y = y[:, :, :, 2, b]

        dsc = dice_loss(_y_pred, _y)  # Use the adjusted dice_loss function
        loss += dsc
    end

    # Average the loss over the batch
    return loss / size(y, 5), y_pred_sigmoid, st
end

# ╔═╡ 402ba194-350e-4ff3-832b-6651be1d9ce7
dev = gpu_device()

# ╔═╡ 6ec3e34b-1c57-4cfb-a50d-ee786c2e4559
begin
	ps, st = Lux.setup(rng, model)
	ps, st = ps |> dev, st |> dev
end

# ╔═╡ b7561ff5-d704-4301-b038-c02bbba91ae2
md"""
## Training Loop
"""

# ╔═╡ 1e79232f-bda2-459a-bc03-85cd8afab3bf
function train_model(model, ps, st, train_loader, val_loader, num_epochs, dev)
    opt_state = create_optimiser(ps)

    # Initialize DataFrame to store metrics
    metrics_df = DataFrame(
        "Epoch" => Int[], 
        "Train_Loss" => Float64[],
        "Validation_Loss" => Float64[], 
        "Dice_Metric" => Float64[],
        "Hausdorff_Metric" => Float64[],
        "Epoch_Duration" => String[]
    )

    best_val_loss = Inf  # Initialize best validation loss to infinity
    best_ps = ps  # Initialize best parameters
    best_st = st  # Initialize best states
    best_epoch = 0  # Initialize best epoch

    for epoch in 1:num_epochs
        @info "Epoch: $epoch"

        # Start timing the epoch
        epoch_start_time = now()

		# Training Phase
		num_batches_train = 0
		total_loss = 0.0
        for (x, y) in train_loader
			num_batches_train += 1
			@info "Step: $num_batches_train"
			x, y = x |> dev, y |> dev

			(loss, y_pred, st), back = pullback(compute_loss, x, y, model, ps, st)
			total_loss += loss
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = update(opt_state, ps, gs)

        end

		# Calculate and log time taken for the epoch
        epoch_duration = now() - epoch_start_time
		
		avg_train_loss = total_loss / num_batches_train
		@info "avg_train_loss: $avg_train_loss"

		if epoch % 5 == 0
			# Validation Phase
	        val_loss = 0.0
	        total_dice = 0.0
	        total_hausdorff = 0.0
	        num_batches = 0
	        num_images = 0
	        ignore_derivatives() do
				for (x, y) in val_loader
					num_batches += 1
				    x, y = x |> dev, y |> dev
				    (loss, y_pred, st) = compute_loss(x, y, model, ps, st)
				    val_loss += loss
					
				    # Process batch for metrics
					y_pred_cpu, y_cpu = y_pred |> cpu_device(), y |> cpu_device()
				    for b in axes(y_cpu, 5)
						num_images += 1
						
				        _y_pred = Bool.(round.(y_pred_cpu[:, :, :, 2, b]))
				        _y = Bool.(y_cpu[:, :, :, 2, b])
				
				        total_dice += dice_metric(_y_pred, _y)
				        total_hausdorff += hausdorff_metric(_y_pred, _y)
				    end
				    
				end
			end
			
			# Calculate average metrics
	        avg_val_loss = val_loss / num_batches
	        avg_dice = total_dice / num_images
	        avg_hausdorff = total_hausdorff / num_images
	        @info "avg_val_loss: $avg_val_loss"
	        @info "avg_dice: $avg_dice"
	        @info "avg_hausdorff: $avg_hausdorff"
	
	        # Check if the current validation loss is better than the best validation loss
	        if avg_val_loss < best_val_loss
	            best_val_loss = avg_val_loss
	            best_ps = ps
	            best_st = st
	            best_epoch = epoch
	        end
	
	        # Append metrics to the DataFrame
			push!(metrics_df, [epoch, avg_train_loss, avg_val_loss, avg_dice, avg_hausdorff, string(epoch_duration)])
	
	        # Write DataFrame to CSV file
	        write("img_seg_metrics.csv", metrics_df)
	
	        @info "Metrics logged for Epoch $epoch"
		end
    end

    # Save the best model
	best_ps = best_ps |> Lux.cpu_device()
	best_st = best_st |> Lux.cpu_device()
    jldsave("params_img_seg_best.jld2"; best_ps)
    jldsave("states_img_seg_best.jld2"; best_st)
    @info "Best model saved from Epoch $best_epoch"

    return best_ps, best_st
end

# ╔═╡ a2e88851-227a-4719-8828-6064f9d3ef81
num_epochs = 100

# ╔═╡ 5cae73af-471c-4068-b9ff-5bc03dd0472d
# ╠═╡ disabled = true
#=╠═╡
ps_final, st_final = train_model(model, ps, st, train_loader, val_loader, num_epochs, dev);
  ╠═╡ =#

# ╔═╡ 7b9b554e-2999-4c57-805e-7bc0d7a0b4e7
#=╠═╡
jldsave("params_img_seg_final.jld2"; ps_final)
  ╠═╡ =#

# ╔═╡ 6432d227-3ff6-4230-9f52-c3e57ba78618
#=╠═╡
jldsave("states_img_seg_final.jld2"; st_final)
  ╠═╡ =#

# ╔═╡ 0dee7c0e-c239-49a4-93c9-5a856b3da883
md"""
## Visualize Training
"""

# ╔═╡ 0bf3a26a-9e18-43d0-b059-d37e8f2e3645
df = read("img_seg_metrics.csv", DataFrame)

# ╔═╡ bc72bff8-a4a8-4736-9aa2-0e87eed243ba
let
	f = Figure()
	ax = Axis(
		f[1, 1:2],
		title = "Losses"
	)
	
	lines!(df[!, :Epoch], df[!, :Train_Loss], label = "Train Loss")
	lines!(df[!, :Epoch], df[!, :Validation_Loss], label = "Validation Loss")

	ylims!(low = 0, high = 1.2)
	axislegend(ax; position = :rt)

	ax = Axis(
		f[2, 1],
		title = "Dice Metric"
	)
	lines!(df[!, :Epoch], df[!, :Dice_Metric], label = "Dice Metric", color = "blue")

	axislegend(ax; position = :rb)

	ax = Axis(
		f[2, 2],
		title = "Hausdorff Metric"
	)
	lines!(df[!, :Epoch], df[!, :Hausdorff_Metric], label = "Hausdorff Metric", color = "green")

	axislegend(ax; position = :rt)

	
	f
end

# ╔═╡ 9a65ff10-649e-4bd7-b079-35fb77eccf53
function model_vis_prep(model, ps_eval, st_eval, transformed_data, dev)
    # Ensure that `xvals` and `yvals` are also on the specified device
    xvals, yvals = getobs(transformed_data, 1)
    xvals = reshape(xvals, (size(xvals)..., 1)) |> dev
    yvals = reshape(yvals, (size(yvals)..., 1)) |> dev

    # Move the model parameters to the specified device
    ps_eval = ps_eval |> dev
    st_eval = st_eval |> dev

    # Evaluate the model
    y_preds, _ = Lux.apply(model, xvals, ps_eval, Lux.testmode(st_eval))
    y_preds = round.(sigmoid.(y_preds))

    # Return the necessary components for the figure
	return xvals |> Lux.cpu_device(), yvals |> Lux.cpu_device(), y_preds |> Lux.cpu_device()
end

# ╔═╡ 61876f59-ea57-4782-82f7-6b292f8e4493
# begin
# 	ps_eval = load("params_img_seg_best.jld2", "best_ps")
# 	st_eval = load("states_img_seg_best.jld2", "best_st")
# end

# ╔═╡ f408f49c-e876-47cd-9bf3-c84f28b84e1f
xvals, yvals, y_preds = model_vis_prep(model, ps_eval, st_eval, transformed_data, dev)

# ╔═╡ c93583ba-9f12-4ea3-9ce5-869443a43c93
md"""
Batch: $(@bind b Slider(axes(xvals, 5); show_value = true))

Z Slice: $(@bind z Slider(axes(yvals, 3); show_value = true, default = div(size(xvals, 3), 2)))
"""

# ╔═╡ 9f6f7552-eeb1-4abd-946c-0b2c57ba7ddf
let
	f = Figure()
	ax = Axis(
		f[1, 1],
		title = "Ground Truth"
	)
	heatmap!(xvals[:, :, z, 1, b], colormap = :grays)
	heatmap!(yvals[:, :, z, 2, b], colormap = (:jet, 0.5))

	ax = Axis(
		f[1, 2],
		title = "Predicted"
	)
	heatmap!(xvals[:, :, z, 1, b], colormap = :grays)
	heatmap!(y_preds[:, :, z, 2, b], colormap = (:jet, 0.5))
	f
end

# ╔═╡ 33b4df0d-86e0-4728-a3bc-928c4dff1400
md"""
# 4. Model Evaluation
"""

# ╔═╡ edddcb37-ac27-4c6a-a98e-c34525cce108
md"""
## Load Test Images
"""

# ╔═╡ 7c821e74-cab5-4e5b-92bc-0e8f76d36556
test_data = HeartSegmentationDataset(data_dir; is_test = true)

# ╔═╡ 6dafe561-411a-45b9-b0ee-d385136e1568
function preprocess_test_data(image, target_size)
    resized_image = resize_image(image, target_size)
    processed_image = Float32.(reshape(resized_image, size(resized_image)..., 1))
    return processed_image
end

# ╔═╡ fe2cfe67-9d87-4eb7-a3d6-13402afbb99a
# ╠═╡ disabled = true
#=╠═╡
transformed_test_data = mapobs(
    x -> preprocess_test_data(x, target_size),
    test_data
)
  ╠═╡ =#

# ╔═╡ bf325c7f-d43a-4a02-b339-2a84eac1c4ff
#=╠═╡
test_loader = DataLoader(transformed_test_data; batchsize = 10, collate = true)
  ╠═╡ =#

# ╔═╡ b206b46a-4261-4727-a4d6-23a305382374
md"""
## Load Best Model
"""

# ╔═╡ 27360e10-ad7e-4fdc-95c5-fef0c5b550dd
md"""
## Predict
"""

# ╔═╡ 13303866-8a40-4325-9334-6de60a2068cd
# ╠═╡ disabled = true
#=╠═╡
begin
	image_test1, image_test2 = getobs(transformed_test_data, 1), getobs(transformed_test_data, 2)
	image_test = cat(image_test1, image_test2, dims = 5)
end;
  ╠═╡ =#

# ╔═╡ 86af32ff-5ffe-4ae4-89ca-89e1165d752c
# ╠═╡ disabled = true
#=╠═╡
begin
	y_test, _ = Lux.apply(model, image_test, ps_eval, Lux.testmode(st_eval))
	y_test = round.(sigmoid.(y_test))
end;
  ╠═╡ =#

# ╔═╡ 1adace71-2b22-461e-86c5-fe42f7b69958
#=╠═╡
typeof(image_test)
  ╠═╡ =#

# ╔═╡ 3545de13-f283-4431-81e7-3abfa14774de
md"""
## Visualize
"""

# ╔═╡ 648e8a2e-0fea-4ee3-8902-eabb79d70d85
#=╠═╡
md"""
Batch: $(@bind b_test Slider(axes(image_test, 5); show_value = true))

Z Slice: $(@bind z_test Slider(axes(image_test, 3); show_value = true, default = div(size(image_test, 3), 2)))
"""
  ╠═╡ =#

# ╔═╡ 2c63c5ff-f364-4f78-bd3c-ac89f32d7b0f
#=╠═╡
let
	f = Figure(size = (700, 500))
	ax = Axis(
		f[1, 1],
		title = "Test Image"
	)
	heatmap!(image_test[:, :, z_test, 1, b_test]; colormap = :grays)

	ax = Axis(
		f[1, 2],
		title = "Segmentation"
	)
	heatmap!(image_test[:, :, z_test, 1, b_test]; colormap = :grays)
	heatmap!(y_test[:, :, z_test, 2, b_test]; colormap = (:jet, 0.5))
	f
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─65dac38d-f955-4058-b577-827d7f8b3db4
# ╟─7cf78ac3-cedd-479d-bc50-769f7b772060
# ╟─fc917017-4d02-4c2d-84d6-b5497d825fff
# ╠═51e6af07-a448-400c-9073-1a7b2c0d69c8
# ╠═8d4a6d5a-c437-43bb-a3db-ab961b218c2e
# ╠═c8d6553a-90df-4aeb-aa6d-a213e16fab48
# ╟─af798f6b-7549-4253-b02b-2ed20dc1125b
# ╠═83b95cee-90ed-4522-b9a8-79c082fce02e
# ╠═af50e5f3-1a1c-47e5-a461-ffbee0329309
# ╟─f0e64ba5-5e11-4ddb-91d3-2a34c60dc6bf
# ╟─ec7734c3-33a5-43c7-82db-2db4dbdc9587
# ╠═7353b7ce-8b33-4602-aed7-2aa24864aca5
# ╠═de5efc37-db19-440e-9487-9a7bea84996d
# ╠═cdfd2412-897d-4642-bb69-f8031c418446
# ╠═b1516500-ad83-41d2-8a1d-093cd0d948e3
# ╠═3e896957-61d8-4750-89bd-be02383417ec
# ╠═4e715848-611a-4125-8ee6-ac5b4d3e4147
# ╠═99211382-7de9-4e97-872f-d0c01b8f8307
# ╠═db2ccf3a-437a-4dfa-ad05-2526c0e2bde0
# ╠═562b3772-89cc-4390-87c3-e7260c8aa86b
# ╠═3ab44a2a-692f-4603-a5a8-81f1d260c13e
# ╠═6d34b756-4da8-427c-91f5-dfb022c4e715
# ╠═9577b91b-faa4-4fc5-9ec2-ed8ca94f2afe
# ╟─0e820544-dc33-43fb-85be-f928758b8b67
# ╠═cf1b6b00-d55c-4310-b1e6-ca03a009a098
# ╠═da9cada1-7ea0-4b6b-a338-d8e08b668d28
# ╠═b4d6f9da-677d-494f-bbc7-811eb65a6bd7
# ╠═3e1b60c0-ea32-4024-b6f7-44d3257a44ac
# ╠═c5bc3f84-8679-4ff7-863e-67d6125e1a4f
# ╠═ea0fd7c2-7cbe-4e30-905e-457ec81b42c5
# ╠═43cf82c5-f0ef-42dd-ad5c-6265d345da9e
# ╠═f48d5547-a80c-4709-aa2c-0dd4a5b2d2a7
# ╠═733e6868-6bd4-4b4a-b1a5-815db1cd8286
# ╟─8d97c2b5-659f-42d8-a86b-00638790b62f
# ╠═6f6e49fc-3322-4da7-b6ff-8846260139b2
# ╠═8e5d073b-98ff-412e-b9fe-70e6e9e912f4
# ╠═5996e996-5d79-48a4-90de-2b07d9b5d59e
# ╠═28592dec-220c-46c5-9e7f-51774e134ff1
# ╠═f191b8dc-5ba5-431e-a46a-cf5109d6fc7b
# ╠═c1d624c8-4e1b-44d3-8f8d-dce740841a20
# ╠═274f277b-0bda-47e7-a52a-e75be9538957
# ╟─03bab55a-6e5e-4b9f-b56a-7e9f993576eb
# ╠═cf23fca5-78f6-4bc4-9f9b-24c062254a58
# ╠═2032b7e6-ceb7-4c08-9b0d-bc704f5e4104
# ╟─2ec43028-c1ab-4df7-9cfe-cc1a4919a7cf
# ╠═8e2f2c6d-127d-42a6-9906-970c09a22e61
# ╟─a6316144-c809-4d2a-bda1-d5128dcf89d3
# ╠═f8fc2cee-c1bd-477d-9595-9427e8764bd6
# ╟─7cb986f8-b338-4046-b569-493e443a8dcb
# ╟─d7e75a72-8281-432c-abab-c254f8c94d3c
# ╟─9dc89870-3d99-472e-8974-712e34a3a789
# ╠═0f5d7796-2c3d-4b74-86c1-a1d4e3922011
# ╠═51e9e7d9-a1d2-4fd1-bdad-52851d9498a6
# ╠═803d918a-66ce-4ed3-a33f-5dda2dd7288e
# ╟─6e2bfcfb-77e3-4532-a14d-10f4b91f2f54
# ╟─bae79c05-034a-4c39-801a-01229b618e94
# ╟─95ad5275-63ca-4f2a-9f3e-6c6a340f5cd4
# ╠═a3f44d7c-efa3-41d0-9509-b099ab7f09d4
# ╠═317c1571-d232-4cab-ac10-9fc3b7ad33b0
# ╟─773aace6-14ad-46f6-a1a6-692247231e90
# ╠═1588d84a-c5f7-4be6-9295-c3594d77b08f
# ╠═f9b0aa7f-d660-4d6f-bd5d-721e5c809b13
# ╠═e682f461-43d7-492a-85a9-2d46e829a125
# ╟─36ad66d6-c484-4073-bf01-1f7ec7012373
# ╠═f55e3c0f-6abe-423c-8319-96146f30eebd
# ╠═bbdaf5c5-9faa-4b61-afab-c0242b8ca034
# ╟─df2dd9a7-045c-44a5-a62c-8d9f2541dc14
# ╟─7cde37c8-4c59-4583-8995-2b01eda95cb3
# ╠═12d42392-ad7b-4c5f-baf5-1f2c6052669e
# ╠═0390bcf5-4cd6-49ba-860a-6f94f8ba6ded
# ╟─a25bdfe6-b24d-446b-926f-6e0727d647a2
# ╠═a6669580-de24-4111-a7cb-26d3e727a12e
# ╠═70bc36db-9ee3-4e1d-992d-abbf55c52070
# ╠═08f2911c-90e7-418e-b9f2-a0722a857bf1
# ╠═402ba194-350e-4ff3-832b-6651be1d9ce7
# ╠═6ec3e34b-1c57-4cfb-a50d-ee786c2e4559
# ╟─b7561ff5-d704-4301-b038-c02bbba91ae2
# ╠═2f6f0755-d71f-4239-a72b-88a545ba8ca1
# ╠═69880e6d-162a-4aae-94eb-103bd35ac3c9
# ╠═dce913e0-126d-4aa3-933a-4f07eea1b8ae
# ╠═c283f9a3-6a76-4186-859f-21cd9efc131f
# ╠═dfc9377a-7cc1-43ba-bb43-683d24e67d79
# ╠═e457a411-2e7b-43b3-a247-23eff94222b0
# ╠═1b5ae165-1069-4638-829a-471b907cce86
# ╠═b04c696b-b404-4976-bfc1-51889ef1d60f
# ╠═1e79232f-bda2-459a-bc03-85cd8afab3bf
# ╠═a2e88851-227a-4719-8828-6064f9d3ef81
# ╠═5cae73af-471c-4068-b9ff-5bc03dd0472d
# ╠═7b9b554e-2999-4c57-805e-7bc0d7a0b4e7
# ╠═6432d227-3ff6-4230-9f52-c3e57ba78618
# ╟─0dee7c0e-c239-49a4-93c9-5a856b3da883
# ╠═c4824b83-01aa-411d-b088-1e5320224e3c
# ╠═0bf3a26a-9e18-43d0-b059-d37e8f2e3645
# ╠═3d4f7938-f7f6-47f1-ad1d-c56a7d7a987f
# ╟─bc72bff8-a4a8-4736-9aa2-0e87eed243ba
# ╠═9a65ff10-649e-4bd7-b079-35fb77eccf53
# ╠═00ea61c1-7d20-4c98-892e-dcdec3b0b43f
# ╠═61876f59-ea57-4782-82f7-6b292f8e4493
# ╠═f408f49c-e876-47cd-9bf3-c84f28b84e1f
# ╟─c93583ba-9f12-4ea3-9ce5-869443a43c93
# ╟─9f6f7552-eeb1-4abd-946c-0b2c57ba7ddf
# ╟─33b4df0d-86e0-4728-a3bc-928c4dff1400
# ╟─edddcb37-ac27-4c6a-a98e-c34525cce108
# ╠═7c821e74-cab5-4e5b-92bc-0e8f76d36556
# ╠═6dafe561-411a-45b9-b0ee-d385136e1568
# ╠═fe2cfe67-9d87-4eb7-a3d6-13402afbb99a
# ╠═bf325c7f-d43a-4a02-b339-2a84eac1c4ff
# ╟─b206b46a-4261-4727-a4d6-23a305382374
# ╟─27360e10-ad7e-4fdc-95c5-fef0c5b550dd
# ╠═13303866-8a40-4325-9334-6de60a2068cd
# ╠═86af32ff-5ffe-4ae4-89ca-89e1165d752c
# ╠═1adace71-2b22-461e-86c5-fe42f7b69958
# ╟─3545de13-f283-4431-81e7-3abfa14774de
# ╟─648e8a2e-0fea-4ee3-8902-eabb79d70d85
# ╟─2c63c5ff-f364-4f78-bd3c-ac89f32d7b0f
