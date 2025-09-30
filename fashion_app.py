import streamlit as st
from src.models.fashion_recommender import FashionRecommender
from PIL import Image
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="AI Fashion Recommender",
    page_icon="👗",
    layout="wide"
)

# Initialize session state for recommender
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
    st.session_state.items_loaded = False

# Title
st.title("👗 AI Fashion Recommender")
st.markdown("Find perfect outfit combinations using AI")

# Sidebar - Upload wardrobe
with st.sidebar:
    st.header("📁 Your Wardrobe")
    
    # Option 1: Upload folder path
    folder_path = st.text_input(
        "Folder path to your clothing photos:",
        value="",
        placeholder="/path/to/your/closet"
    )
    
    # Option 2: Upload individual files
    uploaded_files = st.file_uploader(
        "Or upload photos directly:",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    load_button = st.button("Load Wardrobe", type="primary")
    
    if load_button:
        with st.spinner("Loading your wardrobe... This may take a few minutes..."):
            
            # Initialize recommender
            st.session_state.recommender = FashionRecommender(use_blip2=True)
            
            # Load from folder
            if folder_path and os.path.exists(folder_path):
                image_count = 0
                for image_file in os.listdir(folder_path):
                    if image_file.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(folder_path, image_file)
                        st.session_state.recommender.add_clothing_item(image_path)
                        image_count += 1
                
                st.success(f"Loaded {image_count} items from folder!")
            
            # Load from uploaded files
            elif uploaded_files:
                # Save uploaded files temporarily
                temp_dir = "temp_wardrobe"
                os.makedirs(temp_dir, exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.recommender.add_clothing_item(file_path)
                
                st.success(f"Loaded {len(uploaded_files)} items!")
            
            else:
                st.error("Please provide a folder path or upload files")
                st.stop()
            
            # Initialize outfit combiner
            st.session_state.recommender.initialize_outfit_combiner()
            st.session_state.items_loaded = True
    
    # Show wardrobe summary
    if st.session_state.items_loaded:
        st.markdown("---")
        st.subheader("Wardrobe Summary")
        
        for category, items in st.session_state.recommender.categorized_items.items():
            st.write(f"**{category.upper()}:** {len(items)} items")

# Main content area
if not st.session_state.items_loaded:
    st.info("👈 Upload your wardrobe to get started!")
    st.markdown("""
    ### How to use:
    1. **Upload your clothing photos** via the sidebar
    2. **Enter a search query** (e.g., "casual brunch outfit")
    3. **Get AI-powered outfit recommendations**
    
    ### Features:
    - 🤖 CLIP + BLIP-2 AI analysis
    - 👔 Complete outfit combinations
    - 🎨 Style compatibility scoring
    - 📊 Detailed item metadata
    """)

else:
    # Search interface
    st.header("🔍 Search for Outfits")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "What occasion are you dressing for?",
            placeholder="e.g., casual brunch outfit, formal business meeting, date night...",
            label_visibility="collapsed"
        )
    
    with col2:
        num_results = st.selectbox("Results", [1, 2, 3, 4, 5], index=2)
    
    search_button = st.button("Find Outfits", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Finding perfect outfits..."):
            
            # Get outfit combinations
            combinations = st.session_state.recommender.outfit_combiner.create_outfit_combinations(
                query, top_k=num_results
            )
            
            if not combinations:
                st.warning("No outfit combinations found. Try a different search term.")
            else:
                st.success(f"Found {len(combinations)} outfit combinations!")
                
                # Display each outfit
                for i, combo in enumerate(combinations, 1):
                    st.markdown("---")
                    
                    # Outfit header
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.subheader(f"Outfit #{i}")
                    with col2:
                        st.metric("Overall Score", f"{combo['overall_score']:.2f}")
                    with col3:
                        st.metric("Compatibility", f"{combo['compatibility_score']:.2f}")
                    
                    # Display items in outfit
                    outfit_items = list(combo['outfit'].items())
                    cols = st.columns(len(outfit_items))
                    
                    for col, (category, item) in zip(cols, outfit_items):
                        with col:
                            # Display image
                            img = Image.open(item['path'])
                            st.image(img, use_container_width=True)
                            
                            # Item details
                            st.caption(f"**{category.upper()}**")
                            st.caption(Path(item['path']).name)
                            
                            # Individual score
                            individual_score = combo['individual_scores'].get(category, 'N/A')
                            if individual_score != 'N/A':
                                st.caption(f"Match: {individual_score:.2f}")
                            
                            # Show BLIP-2 analysis if available
                            if item['path'] in st.session_state.recommender.item_metadata:
                                with st.expander("Details"):
                                    meta = st.session_state.recommender.item_metadata[item['path']]
                                    st.write(f"**Style:** {meta['style']}")
                                    st.write(f"**Color:** {meta['color']}")
                                    st.write(f"**Season:** {meta['season']}")
    
    # Browse wardrobe section
    st.markdown("---")
    st.header("👚 Browse Your Wardrobe")
    
    # Category filter
    categories = list(st.session_state.recommender.categorized_items.keys())
    selected_category = st.selectbox("Filter by category:", ["All"] + categories)
    
    # Display items
    if selected_category == "All":
        all_items = []
        for items in st.session_state.recommender.categorized_items.values():
            all_items.extend(items)
    else:
        all_items = st.session_state.recommender.categorized_items[selected_category]
    
    # Create grid
    cols_per_row = 4
    for i in range(0, len(all_items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, all_items[i:i+cols_per_row]):
            with col:
                img = Image.open(item['path'])
                st.image(img, use_container_width=True)
                st.caption(Path(item['path']).name)
                
                if item['path'] in st.session_state.recommender.item_metadata:
                    meta = st.session_state.recommender.item_metadata[item['path']]
                    st.caption(f"{meta['style']} • {meta['color']}")