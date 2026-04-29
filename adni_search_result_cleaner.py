import pandas as pd

# --- Configuration ---
# Replace this with the exact name of the CSV file you downloaded from ADNI
CSV_FILE = 'adni_search_results.csv'


def clean_adni_data(file_path):
  # 1. Load the data
  df = pd.read_csv(file_path)

  # Note: ADNI sometimes calls the ID column "Image ID" or "Image Data ID".
  # Adjust this variable if your CSV header is slightly different.
  id_col = 'Image ID' if 'Image ID' in df.columns else 'Image Data ID'

  # 2. Filter out raw data entirely
  df = df[df['Type'] == 'Pre-processed']

  # 3. Extract the Best MRIs
  # We look for spatially normalized scans (AC-PC or BEaST) to ensure the
  # brains are sitting at the exact same angle in the 3D matrix.
  mri_mask = (df['Modality'] == 'MRI') & \
             (df['Description'].str.contains('AC-PC|BEaST', case=False,
                                             na=False))

  # Drop duplicates to keep exactly ONE MRI per subject
  df_mri = df[mri_mask].drop_duplicates(subset=['Subject ID'], keep='first')

  # 4. Extract the Best PETs (AV45)
  # We strictly target "Standardized Image and Voxel Size" so your CNN
  # doesn't crash from mismatched tensor dimensions.
  pet_mask = (df['Modality'] == 'PET') & \
             (df['Description'].str.contains('AV45', case=False, na=False)) & \
             (df['Description'].str.contains(
               'Standardized Image and Voxel Size', case=False, na=False))

  # Drop duplicates to keep exactly ONE PET per subject
  df_pet = df[pet_mask].drop_duplicates(subset=['Subject ID'], keep='first')

  # 5. The Intersection (Subjects with BOTH)
  # Merge the two dataframes. This acts as an "AND" filter. It only keeps
  # subjects that survived both the MRI and PET filtering above.
  df_final = pd.merge(df_mri, df_pet, on='Subject ID',
                      suffixes=('_MRI', '_PET'))

  # 6. Extract the Image IDs
  mri_ids = df_final[f'{id_col}_MRI'].astype(str).tolist()
  pet_ids = df_final[f'{id_col}_PET'].astype(str).tolist()

  # Combine them into one long list
  all_target_ids = mri_ids + pet_ids

  # 7. Export the clean IDs to a text file
  output_file = 'adni_clean_image_ids.txt'
  with open(output_file, 'w') as f:
    # ADNI's search bar accepts comma-separated IDs
    f.write(','.join(all_target_ids))

  print(
    f"SUCCESS: Found {len(df_final)} subjects with perfect matching multi-modal data.")
  print(f"Saved {len(all_target_ids)} total Image IDs to '{output_file}'.")


def get_massive_adni_dataset(file_path):
  df = pd.read_csv(file_path)
  id_col = 'Image ID' if 'Image ID' in df.columns else 'Image Data ID'

  # 1. Filter out raw data entirely
  df = df[df['Type'] == 'Pre-processed']

  # 2. Extract Standard MRIs (Loosened Filter)
  # We now accept standard ADNI artifact-corrected structural scans.
  mri_mask = (df['Modality'] == 'MRI') & \
             (df['Description'].str.contains(
               'MPRAGE|MP-RAGE|FSPGR|N3|GradWarp|Scaled', case=False,
               na=False))

  df_mri = df[mri_mask].drop_duplicates(subset=['Subject ID'], keep='first')

  # 3. Extract the Best PETs (AV45)
  pet_mask = (df['Modality'] == 'PET') & \
             (df['Description'].str.contains('AV45', case=False, na=False)) & \
             (df['Description'].str.contains(
               'Standardized Image and Voxel Size', case=False, na=False))

  df_pet = df[pet_mask].drop_duplicates(subset=['Subject ID'], keep='first')

  # 4. The Intersection
  df_final = pd.merge(df_mri, df_pet, on='Subject ID',
                      suffixes=('_MRI', '_PET'))

  # 5. Extract and save IDs
  all_target_ids = df_final[f'{id_col}_MRI'].astype(str).tolist() + df_final[
    f'{id_col}_PET'].astype(str).tolist()

  with open('adni_massive_ids.txt', 'w') as f:
    f.write(','.join(all_target_ids))

  print(f"SUCCESS: Found {len(df_final)} subjects.")
  print(f"Saved {len(all_target_ids)} Image IDs.")


if __name__ == "__main__":
  clean_adni_data(CSV_FILE)
  get_massive_adni_dataset(CSV_FILE)
