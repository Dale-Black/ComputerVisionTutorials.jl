### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> title = "Home"

using Markdown
using InteractiveUtils

# ╔═╡ 4b704955-ea7f-48fc-b756-f2c0d860d60d
# ╠═╡ show_logs = false
using Pkg; Pkg.activate("."); Pkg.instantiate()

# ╔═╡ ee1e03f4-7013-4300-8cd4-1a47272befac
using HTMLStrings: to_html, head, link, script, divv, h1, img, p, span, a, figure, hr

# ╔═╡ dd52d41f-433e-491b-a89c-158becd790ae
using PlutoUI: TableOfContents

# ╔═╡ ba623422-a111-11ee-070a-3fad79960969
md"""
# Motivation

This repository pulls together packages in the Julia ecosystem focused on various aspects of deep learning for medical imaging and provides a comprehensive overview of how each package might fit together in a cohesive way for training a neural network.
"""

# ╔═╡ 02a475d1-9208-4fb4-b321-bce622e8448d
md"""
## Comprehensive Tutorials
"""

# ╔═╡ 43f3c01f-6e32-4144-a122-488129d4374b
md"""
## Training Components
"""

# ╔═╡ f5ca1509-dbf2-40ed-87f6-9ca9ed2e5aba
md"""
# Links
"""

# ╔═╡ 7d32eff6-d12c-46a5-aa8e-fd6c60054564
md"""
## Core Packages
"""

# ╔═╡ 63f41d0a-9f2d-402f-8ecb-0960f14c9622
md"""
## Other packages
"""

# ╔═╡ dcf15c2f-7356-4e43-8a16-339151ae833b
to_html(hr())

# ╔═╡ 14c89d42-0996-4ca9-9d65-8743656e9ffd
TableOfContents()

# ╔═╡ d5b8279a-9df1-41a7-a9af-100a7fb8b44c
data_theme = "light";

# ╔═╡ 4f46e4eb-1640-4c0d-9867-e2002ecb6529
function index_title_card(title::String, subtitle::String, image_url::String; data_theme::String = "pastel", border_color::String = "primary")
	return to_html(
	    divv(
	        head(
				link(:href => "https://cdn.jsdelivr.net/npm/daisyui@3.7.4/dist/full.css", :rel => "stylesheet", :type => "text/css"),
	            script(:src => "https://cdn.tailwindcss.com")
	        ),
			divv(:data_theme => "$data_theme", :class => "card card-bordered flex justify-center items-center border-$border_color text-center w-full dark:text-[#e6e6e6]",
				divv(:class => "card-body flex flex-col justify-center items-center",
					img(:src => "$image_url", :class => "h-24 w-24 md:h-40 md:w-40 rounded-md", :alt => "$title Logo"),
					divv(:class => "text-5xl font-bold bg-gradient-to-r from-accent to-primary inline-block text-transparent bg-clip-text py-10", "$title"),
					p(:class => "card-text text-md font-serif", "$subtitle"
					)
				)
			)
	    )
	)
end;

# ╔═╡ 3f662845-48cb-477e-a3d9-f1f9dfe7f2d3
index_title_card(
	"ComputerVisionTutorials.jl",
	"Practical tutorials for deep learning in the field of computer vision with an emphasis on medical imaging, using Julia.",
	"https://img.freepik.com/free-photo/modern-hospital-machinery-illuminates-blue-mri-scanner-generated-by-ai_188544-44420.jpg?ga=GA1.1.1694943658.1700350224&semt=ais_ai_generated";
	data_theme = data_theme
)

# ╔═╡ 20770955-46bd-4d07-b759-f08604583b01
struct Article
	title::String
	path::String
	image_url::String
end

# ╔═╡ c849343a-dbf9-44ed-954c-35ba82918b96
article_list_components = Article[
	Article("Data Preparation", "components/01_data_preparation.jl", "https://img.freepik.com/premium-photo/stickers-basic-gift-box-open-with-simple-geometric-creative-concept-boxes-gift-design_655090-499328.jpg?ga=GA1.1.1694943658.1700350224&semt=ais_ai_generated"),
	Article("Model Building", "components/02_model_building.jl", "https://img.freepik.com/premium-photo/pile-lego-bricks-with-word-lego-it_822916-171.jpg?ga=GA1.1.1694943658.1700350224&semt=ais_ai_generated"),
	Article("Optimisers and Loss Functions", "components/03_optim_loss.jl", "https://img.freepik.com/free-photo/glowing-sine-waves-create-futuristic-backdrop-design-generated-by-ai_188544-36480.jpg?t=st=1704917724~exp=1704921324~hmac=099b8d531988fe63ccf835b69b70af25479fe869b63fbb659cdbae94f52e8620&w=1800"),
];

# ╔═╡ 421ca566-9e2f-4343-b4f1-b6a8bb6482dc
article_list_tutorials = Article[
	Article("3D Segmentation", "tutorials/3D_segmentation.jl", "https://img.freepik.com/free-photo/realistic-heart-shape-studio_23-2150827358.jpg?ga=GA1.1.1694943658.1700350224&semt=ais_ai_generated"),
];

# ╔═╡ 49dd6dbd-04dd-4c67-86ec-1111f7ab6ebb
article_list_packages = Article[
	Article("Losers.jl", "Losers.jl/index.jl", "https://img.freepik.com/free-vector/low-self-esteem-woman-looking-into-mirror_23-2148714425.jpg?size=626&ext=jpg&ga=GA1.1.1427368820.1695503713&semt=ais"),	
	Article("DistanceTransforms.jl", "DistanceTransforms.jl/index.jl", "https://img.freepik.com/free-vector/global-communication-background-business-network-vector-design_53876-151122.jpg"),
	Article("ComputerVisionMetrics.jl", "ComputerVisionMetrics.jl/index.jl", "https://img.freepik.com/free-photo/colorful-bar-graph-sits-table-with-dark-background_1340-34474.jpg"),	
];

# ╔═╡ 53bb4fb7-54d9-4626-a388-356db88c738e
function article_card(article::Article, color::String; data_theme = "pastel")
    a(:href => article.path, :class => "w-1/2 p-2",
		divv(:data_theme => "$data_theme", :class => "card card-bordered border-$color text-center dark:text-[#e6e6e6]",
			divv(:class => "card-body justify-center items-center",
				p(:class => "card-title", article.title),
				p("Click to open the notebook")
			),
			figure(
				img(:class =>"w-full h-60", :src => article.image_url, :alt => article.title)
			)
        )
    )
end;

# ╔═╡ 24c3e2cf-bcbb-41b2-a608-801344e455f6
to_html(
    divv(:class => "flex flex-wrap justify-center items-start",
        [article_card(article, "accent"; data_theme = data_theme) for article in article_list_tutorials]...
    )
)

# ╔═╡ 3f682f9f-94ac-447b-8066-08d7549888bd
to_html(
    divv(:class => "flex flex-wrap justify-center items-start",
        [article_card(article, "primary"; data_theme = data_theme) for article in article_list_components]...
    )
)

# ╔═╡ 16099593-958f-4670-851b-86e71f3fc1d0
to_html(
    divv(:class => "flex flex-wrap justify-center items-start",
        [article_card(article, "accent"; data_theme = data_theme) for article in article_list_packages]...
    )
)

# ╔═╡ 2b810512-b610-4184-b910-0a1302731eac
struct MiniCard
    titles::Array{String,1}
    paths::Array{String,1}
end;

# ╔═╡ b6614476-2a2d-43b9-8b3b-31f2ae6ca7ec
function mini_card(card::MiniCard; data_theme = "pastel", color = "primary")
    # Determine the width class based on the number of items
    width_class = length(card.titles) == 1 ? "w-full mb-2" : "w-1/2 mb-2"
    
    # Generate the card or cards
    cards_html = [
        divv(
            :data_theme => "$data_theme", 
			:class => "card card-bordered text-center $width_class border-$color",
            divv(:class => "card-body",
                p(:class => "card-title text-xs truncate", title)
            )
        ) for title in card.titles
    ]

    if length(cards_html) > 1
        cards_html = divv(:class => "flex w-full", cards_html...)
    else
        cards_html = cards_html[1]
    end
    
    a(:href => "#", :class => "w-full", cards_html)
end;

# ╔═╡ cfb31880-3a8d-40eb-903c-5836d92b52b8
mini_card_others = [
    MiniCard(["Lux.jl"], ["#"]),
    MiniCard(["MLUtils.jl", "MLDatasets.jl"], ["#", "#"]),
    MiniCard(["NIfTI.jl", "DICOM.jl"], ["#", "#"]),
    MiniCard(["Optimisers.jl", "ParameterSchedulers.jl"], ["#", "#"]),
    MiniCard(["..."], ["#"]),
];

# ╔═╡ f4b092cd-b76c-44d6-99d7-1d989e654ab4
to_html(
    divv(:class => "flex flex-wrap justify-center items-start",
        [mini_card(card; data_theme = data_theme, color = "primary") for card in mini_card_others]...
    )
)

# ╔═╡ Cell order:
# ╟─3f662845-48cb-477e-a3d9-f1f9dfe7f2d3
# ╟─ba623422-a111-11ee-070a-3fad79960969
# ╟─02a475d1-9208-4fb4-b321-bce622e8448d
# ╟─24c3e2cf-bcbb-41b2-a608-801344e455f6
# ╟─43f3c01f-6e32-4144-a122-488129d4374b
# ╟─3f682f9f-94ac-447b-8066-08d7549888bd
# ╟─f5ca1509-dbf2-40ed-87f6-9ca9ed2e5aba
# ╟─7d32eff6-d12c-46a5-aa8e-fd6c60054564
# ╟─16099593-958f-4670-851b-86e71f3fc1d0
# ╟─63f41d0a-9f2d-402f-8ecb-0960f14c9622
# ╟─f4b092cd-b76c-44d6-99d7-1d989e654ab4
# ╟─dcf15c2f-7356-4e43-8a16-339151ae833b
# ╟─4b704955-ea7f-48fc-b756-f2c0d860d60d
# ╟─ee1e03f4-7013-4300-8cd4-1a47272befac
# ╟─dd52d41f-433e-491b-a89c-158becd790ae
# ╟─14c89d42-0996-4ca9-9d65-8743656e9ffd
# ╟─d5b8279a-9df1-41a7-a9af-100a7fb8b44c
# ╟─4f46e4eb-1640-4c0d-9867-e2002ecb6529
# ╟─20770955-46bd-4d07-b759-f08604583b01
# ╟─c849343a-dbf9-44ed-954c-35ba82918b96
# ╟─421ca566-9e2f-4343-b4f1-b6a8bb6482dc
# ╟─49dd6dbd-04dd-4c67-86ec-1111f7ab6ebb
# ╟─53bb4fb7-54d9-4626-a388-356db88c738e
# ╟─2b810512-b610-4184-b910-0a1302731eac
# ╟─b6614476-2a2d-43b9-8b3b-31f2ae6ca7ec
# ╟─cfb31880-3a8d-40eb-903c-5836d92b52b8
