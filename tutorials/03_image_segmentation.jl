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
	Pluto automatically handles the install of all of the packages imported throughout the notebook. You might notice that the first time running this notebook takes a while to get started, this is likely because all of these packages are being installed.md

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
ps_final, st_final = train_model(model, ps, st, train_loader, val_loader, num_epochs, dev);

# ╔═╡ 7b9b554e-2999-4c57-805e-7bc0d7a0b4e7
jldsave("params_img_seg_final.jld2"; ps_final)

# ╔═╡ 6432d227-3ff6-4230-9f52-c3e57ba78618
jldsave("states_img_seg_final.jld2"; st_final)

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
transformed_test_data = mapobs(
    x -> preprocess_test_data(x, target_size),
    test_data
)

# ╔═╡ bf325c7f-d43a-4a02-b339-2a84eac1c4ff
test_loader = DataLoader(transformed_test_data; batchsize = 10, collate = true)

# ╔═╡ b206b46a-4261-4727-a4d6-23a305382374
md"""
## Load Best Model
"""

# ╔═╡ 27360e10-ad7e-4fdc-95c5-fef0c5b550dd
md"""
## Predict
"""

# ╔═╡ 13303866-8a40-4325-9334-6de60a2068cd
begin
	image_test1, image_test2 = getobs(transformed_test_data, 1), getobs(transformed_test_data, 2)
	image_test = cat(image_test1, image_test2, dims = 5)
end;

# ╔═╡ 86af32ff-5ffe-4ae4-89ca-89e1165d752c
# ╠═╡ disabled = true
#=╠═╡
begin
	y_test, _ = Lux.apply(model, image_test, ps_eval, Lux.testmode(st_eval))
	y_test = round.(sigmoid.(y_test))
end;
  ╠═╡ =#

# ╔═╡ 1adace71-2b22-461e-86c5-fe42f7b69958
typeof(image_test)

# ╔═╡ 3545de13-f283-4431-81e7-3abfa14774de
md"""
## Visualize
"""

# ╔═╡ 648e8a2e-0fea-4ee3-8902-eabb79d70d85
md"""
Batch: $(@bind b_test Slider(axes(image_test, 5); show_value = true))

Z Slice: $(@bind z_test Slider(axes(image_test, 3); show_value = true, default = div(size(image_test, 3), 2)))
"""

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
ComputerVisionMetrics = "56beca70-ca20-45da-83c4-a042539b6c19"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
DistanceTransforms = "71182807-4d06-4237-8dd0-bdafe4d097e2"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
ImageFiltering = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
ImageTransformations = "02fcd773-0e25-5acc-982a-7f6622650795"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
Losers = "1785af8d-d312-496e-9b53-daf6ddaba92c"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
NIfTI = "a3a9e032-41b5-5fc4-967a-a6b7a19844d3"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Tar = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
CSV = "~0.10.13"
CairoMakie = "~0.11.9"
ChainRulesCore = "~1.23.0"
ComputerVisionMetrics = "~0.1.0"
DataFrames = "~1.6.1"
DistanceTransforms = "~0.2.1"
FileIO = "~1.16.2"
Glob = "~1.3.1"
HTTP = "~1.10.3"
ImageFiltering = "~0.7.8"
ImageTransformations = "~0.10.1"
JLD2 = "~0.4.46"
Losers = "~0.2.0"
Lux = "~0.5.23"
LuxCUDA = "~0.3.2"
MLUtils = "~0.4.4"
NIfTI = "~0.6.0"
Optimisers = "~0.3.2"
PlutoUI = "~0.7.58"
Zygote = "~0.6.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "bb9b4b280c33cbf6d2afd7998907ff8481b388bf"

[[deps.ADTypes]]
git-tree-sha1 = "016833eb52ba2d6bea9fcb50ca295980e728ee24"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.7"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractLattices]]
git-tree-sha1 = "222ee9e50b98f51b5d78feb93dd928880df35f06"
uuid = "398f06c4-4d28-53ec-89ca-5b2656b7603d"
version = "0.3.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cea4ac3f5b4bc4b3000aa55afb6e5626518948fa"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.3"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "44691067188f6bd1b2289552a23e4b7572f4528d"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.9.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "588e0d680ad1d7201d4c6a804dcb1cd9cba79fbb"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.3"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "7aa7ad1682f3d5754e3491bb59b8103cae28e3a3"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.40"

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

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "a44910ceb69b0d44fe262dd451ab11ead3ed0be8"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.13"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics"]
git-tree-sha1 = "baa8ea7a1ea63316fa3feb454635215773c9c845"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.2.0"
weakdeps = ["ChainRulesCore", "SpecialFunctions"]

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "d01bfc999768f0a31ed36f5d22a76161fc63079c"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.7.0+1"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "2cb12f6b2209f40a4b8967697689a47c50485490"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.3"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e25c009d2bf16c2c31a70a6e9e8939f7325cc84"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.11.1+0"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "75923dce4275ead3799b238e10178a68c07dbd3b"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "8.9.4+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "6dc1bbdd6a133adf4aa751d12dbc2c6ae59f873d"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.11.9"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a4c43f59baa34011e303e76f5c8c91bf58415aaf"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "4e42872be98fa3343c4f8458cbda8c5c6a6fa97c"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.63.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "575cd02e080939a33b6df6c5853d14924c08e35b"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.23.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "9b1ca1aa6ce3f71b3d1840c538a8210a043625eb"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

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

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ComputerVisionMetrics]]
deps = ["ImageMorphology", "StatsBase"]
git-tree-sha1 = "523738766a10afbfba028d48981199cade5e531f"
uuid = "56beca70-ca20-45da-83c4-a042539b6c19"
version = "0.1.0"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "0f4b5d62a88d8f59003e43c25a8a90de9eb76317"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.18"

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

[[deps.DelaunayTriangulation]]
deps = ["DataStructures", "EnumX", "ExactPredicates", "Random", "SimpleGraphs"]
git-tree-sha1 = "d4e9dc4c6106b8d44e40cd4faf8261a678552c7c"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "0.8.12"

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

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ab3f7e1819dba9434a3a5126510c8fda3a4e7000"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.1+0"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

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

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "c5c28c245101bd59154f649e19b038d15901b5dc"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.2"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "bc0c5092d6caaea112d3c8e3b238d61563c58d5f"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.23.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Format]]
git-tree-sha1 = "f3cf88025f6d03c194d73f5d13fee9004a108329"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.6"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "055626e1a35f6771fe99060e835b72ca61a52621"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8ae30e786837ce0a24f5e2186938bf3251ab94b2"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "47e4686ec18a9620850bad110b79966132f14283"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "10.0.2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "a846f297ce9d09ccba02ead0cae70690e072a119"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.25.0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "d4f85701f569584f2cff7ba67a137d03f0cfb7d0"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.3"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "5694b56ccf9d15addedc35e9a4ba9c317721b788"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.10"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "359a1ba2e320790ddbe4ee8b4d54a305c0ea2aff"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.0+0"

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "af13a277efd8a6e716d79ef635d5342ccb75be61"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.10.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "db864f2d91f68a5912937af80327d288ea1f3aee"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.3"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "5d8c5713f38f7bc029e26627b687710ba406d0dd"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.12"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "432ae2b430a18c58eb7eca9ef8d0f2db90bc749c"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.8"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e0884bdf01bbbb111aea77c348368a86fb4b5ab6"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.1"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5fdf2fe6724d8caabf43b557b84ce53f3b7e2f6b"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.0.2+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "RoundingEmulator"]
git-tree-sha1 = "2d6d22fe481eff6e337808cc0880c567d7324f9a"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.8"
weakdeps = ["DiffRules", "ForwardDiff", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "5ea6acdd53a51d897672edb694e3cc2912f3f8a7"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.46"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

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
git-tree-sha1 = "ed7167240f40e62d97c1f5f7735dea6de3cc5c49"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.18"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "7c6650580b4c3169d9905858160db895bff6d2e2"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.6.1"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "88b916503aac4fb7f701bb625cd84ca5dd1677bc"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.29+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

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

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dae976433497a2f841baadea93d27e68f1a12a97"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.39.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0a04a1318df1bf510beb2562cf90fb0c386f58c4"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.39.3+1"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "3a994404d3f6709610701c7dabfc03fed87a81f8"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearAlgebraX]]
deps = ["LinearAlgebra", "Mods", "Primes", "SimplePolynomials"]
git-tree-sha1 = "d76cec8007ec123c2b681269d40f94b053473fcf"
uuid = "9b3f67b0-2d00-526e-9884-9e4938f8fb88"
version = "0.2.7"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

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
git-tree-sha1 = "62a3bc5c6c7cab3fe2cfdcd9b63ca030b5cbecb8"
uuid = "1785af8d-d312-496e-9b53-daf6ddaba92c"
version = "0.2.0"

[[deps.Lux]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "ChainRulesCore", "ConcreteStructs", "ConstructionBase", "Functors", "GPUArraysCore", "LinearAlgebra", "LuxCore", "LuxDeviceUtils", "LuxLib", "MacroTools", "Markdown", "Optimisers", "PrecompileTools", "Random", "Reexport", "Setfield", "SparseArrays", "Statistics", "TruncatedStacktraces", "WeightInitializers"]
git-tree-sha1 = "24416cf0565c5272011dcd93951e7c7fde35e92e"
uuid = "b2108857-7c20-44ae-9111-449ecde12c47"
version = "0.5.23"

    [deps.Lux.extensions]
    LuxChainRulesExt = "ChainRules"
    LuxComponentArraysExt = "ComponentArrays"
    LuxComponentArraysReverseDiffExt = ["ComponentArrays", "ReverseDiff"]
    LuxFluxExt = "Flux"
    LuxLuxAMDGPUExt = "LuxAMDGPU"
    LuxReverseDiffExt = "ReverseDiff"
    LuxSimpleChainsExt = "SimpleChains"
    LuxTrackerExt = "Tracker"
    LuxZygoteExt = "Zygote"

    [deps.Lux.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    LuxAMDGPU = "83120cb1-ca15-4f04-bf3b-6967d2e6b60b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SimpleChains = "de6bee2f-e2f4-4ec7-b6ed-219cc6f6e9e5"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxCUDA]]
deps = ["CUDA", "Reexport", "cuDNN"]
git-tree-sha1 = "7cf839d5eecb41e50dd430a9dcf9398727b71e8a"
uuid = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
version = "0.3.2"

[[deps.LuxCore]]
deps = ["Functors", "Random", "Setfield"]
git-tree-sha1 = "f79fa446211545f0f84206f5e415a1b4eedbb678"
uuid = "bb33d45b-7691-41d6-9220-0943567d0623"
version = "0.1.12"

[[deps.LuxDeviceUtils]]
deps = ["Adapt", "ChainRulesCore", "Functors", "LuxCore", "PrecompileTools", "Preferences", "Random"]
git-tree-sha1 = "fa564c5bd5acd33faeb8636d5a8f049b14ba8759"
uuid = "34f89e08-e1d5-43b4-8944-0b49ac560553"
version = "0.1.17"

    [deps.LuxDeviceUtils.extensions]
    LuxDeviceUtilsFillArraysExt = "FillArrays"
    LuxDeviceUtilsGPUArraysExt = "GPUArrays"
    LuxDeviceUtilsLuxAMDGPUExt = "LuxAMDGPU"
    LuxDeviceUtilsLuxCUDAExt = "LuxCUDA"
    LuxDeviceUtilsMetalGPUArraysExt = ["GPUArrays", "Metal"]
    LuxDeviceUtilsRecursiveArrayToolsExt = "RecursiveArrayTools"
    LuxDeviceUtilsSparseArraysExt = "SparseArrays"
    LuxDeviceUtilsZygoteExt = "Zygote"

    [deps.LuxDeviceUtils.weakdeps]
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    LuxAMDGPU = "83120cb1-ca15-4f04-bf3b-6967d2e6b60b"
    LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxLib]]
deps = ["ChainRulesCore", "KernelAbstractions", "Markdown", "NNlib", "PrecompileTools", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "186650b8b54b80607491b626a8b4a27ff179942c"
uuid = "82251201-b29d-42c6-8e01-566dec8acb11"
version = "0.3.10"

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

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "b45738c2e3d0d402dffa32b2c1654759a2ac35a4"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "StableHashTraits", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "27af6be179c711fb916a597b6644fbb5b80becc0"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.20.8"

[[deps.MakieCore]]
deps = ["Observables", "REPL"]
git-tree-sha1 = "248b7a4be0f92b497f7a331aed02c1e9a878f46b"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.7.3"

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

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "679c1aec6934d322783bd15db4d18f898653be4f"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.27.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "96ca8a313eb6437db5ffe946c457a401bbb8ce1d"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.7"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

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

[[deps.Mods]]
git-tree-sha1 = "924f962b524a71eef7a21dae1e6853817f9b658f"
uuid = "7475f97c-0381-53b1-977b-4c60186c8d62"
version = "2.2.4"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.Multisets]]
git-tree-sha1 = "8d852646862c96e226367ad10c8af56099b4047e"
uuid = "3b2b4ff1-bcff-5658-a3ee-dbcf1ce5ac09"
version = "0.4.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "302fd161eb1c439e4115b51ae456da4e9984f130"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.1"

[[deps.NIfTI]]
deps = ["Base64", "CodecZlib", "MappedArrays", "Mmap", "TranscodingStreams"]
git-tree-sha1 = "21e5b879564607ea98fb680c98a1b7838b7d7f1c"
uuid = "a3a9e032-41b5-5fc4-967a-a6b7a19844d3"
version = "0.6.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "877f15c331337d54cf24c797d5bcb2e48ce21221"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.12"

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
git-tree-sha1 = "53046f0483375e3ed78e49190f1154fa0a4083a1"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "0.3.4"

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

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "6a731f2b5c03157418a20c12195eb4b74c8f8621"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.13.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "af81a32750ebc831ee28bdaaba6e1067decef51e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60e3045590bd104a16fefb12836c00c0ef8c7f8c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "MathOptInterface", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d024bfb56144d947d4fafcd9cb5cafbe3410b133"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.2"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "264b061c1903bc0fe9be77cb9050ebacff66bb63"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.2"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "526f5a03792669e4187e584e8ec9d534248ca765"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.52.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[deps.Permutations]]
deps = ["Combinatorics", "LinearAlgebra", "Random"]
git-tree-sha1 = "eb3f9df2457819bf0a9019bd93cc451697a0751e"
uuid = "2ae35dd2-176d-5d53-8349-f30d82d94d4f"
version = "0.4.20"

[[deps.PikaParser]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "d6ff87de27ff3082131f31a714d25ab6d0a88abf"
uuid = "3bbf5609-3e7b-44cd-8549-7c69f321e792"
version = "0.6.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "a9c7a523d5ed375be3983db190f6a5874ae9286d"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.6"
weakdeps = ["ChainRulesCore", "FFTW", "MakieCore", "MutableArithmetics"]

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "cb420f77dc474d23ee47ca8d14c90810cafe69e7"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.6"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "763a8ceb07833dd51bb9e3bbca372de32c0605ad"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.RingLists]]
deps = ["Random"]
git-tree-sha1 = "f39da63aa6d2d88e0c1bd20ed6a3ff9ea7171ada"
uuid = "286e9d63-9694-5540-9e3c-4e6708fa07b2"
version = "0.2.8"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "2a0a5d8569f481ff8840e3b7c84bbf188db6a3fe"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.0"
weakdeps = ["RecipesBase"]

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

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

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleGraphs]]
deps = ["AbstractLattices", "Combinatorics", "DataStructures", "IterTools", "LightXML", "LinearAlgebra", "LinearAlgebraX", "Optim", "Primes", "Random", "RingLists", "SimplePartitions", "SimplePolynomials", "SimpleRandom", "SparseArrays", "Statistics"]
git-tree-sha1 = "f65caa24a622f985cc341de81d3f9744435d0d0f"
uuid = "55797a34-41de-5266-9ec1-32ac4eb504d3"
version = "0.8.6"

[[deps.SimplePartitions]]
deps = ["AbstractLattices", "DataStructures", "Permutations"]
git-tree-sha1 = "e182b9e5afb194142d4668536345a365ea19363a"
uuid = "ec83eff0-a5b5-5643-ae32-5cbf6eedec9d"
version = "0.3.2"

[[deps.SimplePolynomials]]
deps = ["Mods", "Multisets", "Polynomials", "Primes"]
git-tree-sha1 = "7063828369cafa93f3187b3d0159f05582011405"
uuid = "cc47b68c-3164-5771-a705-2bc0097375a0"
version = "0.2.17"

[[deps.SimpleRandom]]
deps = ["Distributions", "LinearAlgebra", "Random"]
git-tree-sha1 = "3a6fb395e37afab81aeea85bae48a4db5cd7244a"
uuid = "a6525b86-64cd-54fa-8f65-62fc48bdc0e8"
version = "0.3.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

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

[[deps.StableHashTraits]]
deps = ["Compat", "PikaParser", "SHA", "Tables", "TupleTools"]
git-tree-sha1 = "10dc702932fe05a0e09b8e5955f00794ea1e8b12"
uuid = "c5dd0088-6c3f-4803-b00e-f31a60c170fa"
version = "1.1.8"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d2fdac9ff3906e27f7a618d47b676941baa6c80c"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.10"

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
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"
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

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"
weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

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

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

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
git-tree-sha1 = "3064e780dbb8a9296ebb3af8f440f787bb5332af"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.80"

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

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.TupleTools]]
git-tree-sha1 = "41d61b1c545b06279871ef1a4b5fcb2cac2191cd"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.5.0"

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

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WeightInitializers]]
deps = ["ChainRulesCore", "LinearAlgebra", "PartialFunctions", "PrecompileTools", "Random", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "f0e6760ef9d22f043710289ddf29e4a4048c4822"
uuid = "d49dbf32-c5c2-4618-8acc-27bb2598ef2d"
version = "0.1.7"
weakdeps = ["CUDA"]

    [deps.WeightInitializers.extensions]
    WeightInitializersCUDAExt = "CUDA"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "07e470dabc5a6a4254ffebc29a1b3fc01464e105"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "4ddb4470e47b0094c93055a3bcae799165cc68f1"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.69"

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
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDA_Runtime_Discovery", "CUDNN_jll"]
git-tree-sha1 = "d433ec29756895512190cac9c96666d879f07b92"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.3.0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─65dac38d-f955-4058-b577-827d7f8b3db4
# ╟─7cf78ac3-cedd-479d-bc50-769f7b772060
# ╟─fc917017-4d02-4c2d-84d6-b5497d825fff
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
