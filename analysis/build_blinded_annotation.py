from pathlib import Path
import os
import pandas as pd
import numpy as np
import shutil
import hashlib
import html

# Data/artefact root. Set WYSIWYR_DATA_ROOT to the directory that contains
# `wysiwyr_real` (e.g. export WYSIWYR_DATA_ROOT=/path/to/data_root).
ROOT = Path(os.environ.get("WYSIWYR_DATA_ROOT", ".")).expanduser() / "wysiwyr_real"
DATA_ROOT = ROOT / "data" / "TestDataset"

FAILURE_CSV = (
    ROOT /
    "usr_failure_analysis_20260913" /
    "usr_case_level_failure.csv"
)

OUT = ROOT / "difficulty_annotation_20260913"
IMG_OUT = OUT / "blinded_images"
BLIND = OUT / "blinded_package"
PRIVATE = OUT / "private_key"

SEED = 20260913

DATASETS = [
    "CVC-300",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
    "Kvasir",
]

# ============================================================
# 1. Load exactly the 798 Stage-5 cases
# ============================================================
cases = pd.read_csv(FAILURE_CSV)

cases["dataset"] = cases["dataset"].astype(str)
cases["case_id"] = cases["case_id"].astype(str)

cases = (
    cases[["dataset", "case_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print("Cases from Stage 5:", len(cases))

if len(cases) != 798:
    raise RuntimeError(
        f"Expected 798 unique cases, got {len(cases)}"
    )

# ============================================================
# 2. Locate original RGB images
# ============================================================
records = []
missing = []

for _, row in cases.iterrows():

    ds = row["dataset"]
    cid = row["case_id"]

    img_dir = DATA_ROOT / ds / "image"

    candidates = []

    for suffix in [
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff"
    ]:
        p = img_dir / f"{cid}{suffix}"
        if p.exists():
            candidates.append(p)

    # fallback: stem matching
    if not candidates:
        for p in img_dir.iterdir():
            if p.is_file() and p.stem == cid:
                candidates.append(p)

    if len(candidates) == 0:
        missing.append((ds, cid))
        continue

    if len(candidates) > 1:
        print(
            "WARNING multiple matches:",
            ds, cid, candidates
        )

    p = candidates[0]

    records.append({
        "dataset": ds,
        "case_id": cid,
        "source_image": str(p),
        "source_filename": p.name,
    })

df = pd.DataFrame(records)

print("Images located:", len(df))
print("Missing:", len(missing))

if missing:
    print("Missing cases:")
    for x in missing:
        print(x)

if len(df) != 798:
    raise RuntimeError(
        f"Only located {len(df)}/798 images."
    )

# ============================================================
# 3. Randomize cases BEFORE assigning anonymous IDs
# ============================================================
rng = np.random.default_rng(SEED)
order = rng.permutation(len(df))

df = df.iloc[order].reset_index(drop=True)

df["anon_id"] = [
    f"CASE_{i:04d}"
    for i in range(1, len(df) + 1)
]

# ============================================================
# 4. Copy ORIGINAL images with anonymous names
#    No masks / predictions are copied.
# ============================================================
image_files = []
sha256s = []

for _, row in df.iterrows():

    src = Path(row["source_image"])

    # Preserve original file bytes and extension
    dst_name = row["anon_id"] + src.suffix.lower()
    dst = IMG_OUT / dst_name

    shutil.copy2(src, dst)

    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    image_files.append(dst_name)
    sha256s.append(h.hexdigest())

df["anonymous_image"] = image_files
df["sha256"] = sha256s

# ============================================================
# 5. PRIVATE key
#    This file must NEVER be shown during annotation.
# ============================================================
private_cols = [
    "anon_id",
    "dataset",
    "case_id",
    "source_filename",
    "source_image",
    "anonymous_image",
    "sha256",
]

df[private_cols].to_csv(
    PRIVATE /
    "KEY_DO_NOT_SHOW_ANNOTATORS.csv",
    index=False
)

# ============================================================
# 6. Annotation forms
# ============================================================
base_form = pd.DataFrame({
    "anon_id": df["anon_id"],
    "image_file": df["anonymous_image"],

    # 0 = none/minimal
    # 1 = moderate
    # 2 = severe
    # 9 = uncertain
    "reflection": "",

    "mucus_occlusion": "",

    "blur": "",

    # 0 = single/no additional lesion
    # 1 = multiple spatially distinct lesions
    # 9 = uncertain
    "multi_lesion": "",

    # Optional
    "ungradable": "",

    "notes": "",
})

base_form.to_csv(
    BLIND /
    "difficulty_annotation_form.csv",
    index=False
)

# Same cases, independently randomized row order
form_A = base_form.sample(
    frac=1,
    random_state=20260913
).reset_index(drop=True)

form_B = base_form.sample(
    frac=1,
    random_state=20260914
).reset_index(drop=True)

form_A.to_csv(
    BLIND /
    "difficulty_annotation_ANNOTATOR_A.csv",
    index=False
)

form_B.to_csv(
    BLIND /
    "difficulty_annotation_ANNOTATOR_B.csv",
    index=False
)

# ============================================================
# 7. Annotation guide
# ============================================================
guide = r"""
WYSIWYR DIFFICULTY / FAILURE-SCENARIO BLINDED ANNOTATION
========================================================

IMPORTANT
---------
Inspect ORIGINAL RGB images only.

DO NOT inspect:
- ground-truth masks
- Baseline predictions
- USR predictions
- ABLoss predictions
- Both predictions
- Dice / Boundary Dice
- improve / worsen information
- dataset identity

The purpose is to annotate image characteristics independently
of model outcomes.

--------------------------------------------------------
1. REFLECTION
--------------------------------------------------------

0 = none / minimal
    No meaningful specular reflection affecting visual assessment.

1 = moderate
    Reflection is clearly present but does not substantially
    obscure the lesion or relevant boundary.

2 = severe
    Strong specular reflection substantially obscures part of
    the lesion, lesion boundary, or immediately adjacent mucosa.

9 = uncertain

--------------------------------------------------------
2. MUCUS / OCCLUSION
--------------------------------------------------------

0 = none / minimal
    No meaningful mucus, debris, fluid, instrument, or other
    material obscuring the target region.

1 = moderate
    Partial occlusion is present but most of the lesion and
    boundary remain visually assessable.

2 = severe
    Substantial mucus/debris/occlusion obscures a meaningful
    portion of the lesion or boundary.

9 = uncertain

--------------------------------------------------------
3. BLUR
--------------------------------------------------------

0 = none / minimal
    Lesion and boundary are reasonably sharp.

1 = moderate
    Motion or defocus blur is visible, but the lesion remains
    generally assessable.

2 = severe
    Strong blur substantially reduces boundary or lesion
    visibility.

9 = uncertain

--------------------------------------------------------
4. MULTI-LESION
--------------------------------------------------------

0 = no
    One primary localized lesion is visible.

1 = yes
    Two or more spatially distinct lesion-like targets are
    visible in the same field of view.

9 = uncertain

Do NOT mark folds, bubbles, stool, reflection or normal mucosal
structures as separate lesions merely because they form distinct
regions.

--------------------------------------------------------
5. UNGRADABLE
--------------------------------------------------------

0 = gradable
1 = image cannot be reliably evaluated for the above categories

Use sparingly.

--------------------------------------------------------
ANNOTATION PRINCIPLES
--------------------------------------------------------

1. Judge image characteristics, not whether segmentation seems easy.
2. Do not infer model performance.
3. Do not use any mask or prediction.
4. When uncertain between severity 0 and 1, select the closer visual
   category rather than using 9.
5. Use 9 only when the category genuinely cannot be determined.
6. Add brief notes only when necessary.

Recommended:
Two independent annotators.
Resolve disagreements only AFTER both annotation files are completed.
"""

(BLIND / "ANNOTATION_GUIDE.txt").write_text(
    guide.strip() + "\n",
    encoding="utf-8"
)

# ============================================================
# 8. Copy blinded images into package
# ============================================================
package_img = BLIND / "images"
package_img.mkdir(exist_ok=True)

for p in IMG_OUT.iterdir():
    shutil.copy2(p, package_img / p.name)

# ============================================================
# 9. Generate HTML galleries
# ============================================================
PAGE_SIZE = 80

rows = base_form.to_dict("records")

n_pages = (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE

index_links = []

for page_idx in range(n_pages):

    subset = rows[
        page_idx * PAGE_SIZE:
        (page_idx + 1) * PAGE_SIZE
    ]

    page_name = f"gallery_{page_idx+1:02d}.html"

    index_links.append(
        f'<li><a href="{page_name}">'
        f'Gallery {page_idx+1:02d}</a></li>'
    )

    body = []

    for r in subset:

        anon = html.escape(str(r["anon_id"]))
        img = html.escape(str(r["image_file"]))

        body.append(f"""
        <div class="case">
            <div class="id">{anon}</div>
            <img src="images/{img}">
        </div>
        """)

    page = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Blinded Difficulty Annotation</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 20px;
    background: #f5f5f5;
}}

.case {{
    background: white;
    margin-bottom: 28px;
    padding: 14px;
    border: 1px solid #bbb;
}}

.id {{
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
}}

img {{
    max-width: 900px;
    max-height: 700px;
    width: auto;
    height: auto;
}}
</style>
</head>

<body>

<h2>
Blinded Difficulty Annotation —
Gallery {page_idx+1}/{n_pages}
</h2>

<p>
Use the accompanying annotation CSV and ANNOTATION_GUIDE.txt.
Do not inspect predictions or masks.
</p>

{''.join(body)}

</body>
</html>
"""

    (BLIND / page_name).write_text(
        page,
        encoding="utf-8"
    )

index_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WYSIWYR Blinded Annotation</title>
</head>
<body>
<h2>WYSIWYR Difficulty Annotation</h2>

<p>
798 cases are divided into {n_pages} galleries.
Read ANNOTATION_GUIDE.txt before annotation.
</p>

<ul>
{''.join(index_links)}
</ul>

</body>
</html>
"""

(BLIND / "index.html").write_text(
    index_html,
    encoding="utf-8"
)

# ============================================================
# 10. Validation
# ============================================================
blind_images = list(
    (BLIND / "images").glob("*")
)

print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)

print("Cases:", len(df))
print("Blinded images:", len(blind_images))
print("Unique anon IDs:", df["anon_id"].nunique())

print()
print("Per-dataset PRIVATE counts:")
print(df["dataset"].value_counts())

assert len(df) == 798
assert len(blind_images) == 798
assert df["anon_id"].nunique() == 798

# Ensure blinded CSV contains no outcomes
for forbidden in [
    "dice",
    "boundary",
    "usr",
    "worsen",
    "dataset",
    "case_id",
]:
    for col in base_form.columns:
        if forbidden in col.lower():
            raise RuntimeError(
                f"Forbidden outcome/identity field in blind form: {col}"
            )

print()
print("BLINDING VALIDATION PASSED")
print("Package generation completed.")
