# 🎬 Anime Recommendation System

A highly optimized, content-based Anime Recommendation Engine built with Python and Streamlit. Search for any anime you love (by its official title or an English alias), and the system mathematically calculates the top 5 most similar shows based on genres, themes, and tagging data.

## ✨ Features

* **⚡ Lightning Fast Search:** Uses unified dictionary mapping and Pandas `itertuples()` for instantaneous startup and indexing.
* **🧠 Machine Learning Core:** Utilizes Scikit-Learn's `TfidfVectorizer` and `cosine_similarity` to find mathematically overlapping anime tags.
* **💾 Zero Memory Bloat:** Optimized to calculate similarity arrays *on-the-fly*.
* **🎨 Custom UI:** Features a custom CSS aesthetic with frosted-glass cards, cached background images, and CSS line-clamping to perfectly format long Light Novel titles.

---

## 📂 Project Structure

```text
📁 IITG-Term-Project-2
├── 📁 .venv                                 # Python virtual environment
├── 📁 src                                   # Main source code directory
│   ├── 📄 anime-offline-database-minified.json # The dataset
│   ├── 📄 app.py                            # The main Streamlit application
│   └── 🖼️ BG_Wallpaper.jpg                  # UI Background image
└── 📄 README.md                             # Project documentation
```

---

## 📊 Dataset Attribution

This project is powered by the incredibly comprehensive **Anime Offline Database**. 

* **Repository:** [manami-project/anime-offline-database](https://github.com/manami-project/anime-offline-database/tree/master)
* **File Used:** `anime-offline-database-minified.json`

> **Note:** Due to GitHub file size limits, you may need to download the JSON dataset directly from the Manami Project repository and place it inside your local `src/` folder before running the application.

---

## 🚀 Installation & Setup

Follow these steps to get the project running on your local machine:

**1. Clone the repository**
```bash
git clone https://github.com/arpitraj19792/IITG-Term-Project-2
cd IITG-Term-Project-2
```

**2. Create and activate a Virtual Environment**
```bash
# For Windows
python -m venv .venv
.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install streamlit pandas scikit-learn
```

**4. Download the Dataset**
Ensure that your downloaded `anime-offline-database-minified.json` file is placed directly inside the `src/` directory.

**5. Run the Application**
```bash
streamlit run src/app.py
```

---

## 🛠️ Built With

* **[Streamlit](https://streamlit.io/):** For the interactive web interface and custom frosted-glass UI rendering.
* **[Pandas](https://pandas.pydata.org/):** For high-speed data manipulation, lightweight C-tuple iteration, and dictionary mapping.
* **[Scikit-Learn](https://scikit-learn.org/):** For text-based TF-IDF vectorization and rapid, on-the-fly cosine similarity calculations.
