import os
import argparse
import numpy as np
import ants
from pathlib import Path


def batch_harmonize(input_dir, output_dir, output_shape=(128, 128, 128)):
  input_path = Path(input_dir)
  output_path = Path(output_dir)

  output_path.mkdir(parents=True, exist_ok=True)

  print("Loading MNI152 Atlas Template...")
  template = ants.image_read(ants.get_ants_data('mni'))

  subject_dirs = [d for d in input_path.iterdir() if d.is_dir()]
  total_subjects = len(subject_dirs)

  print(f"Found {total_subjects} subjects. Starting batch harmonization...\n")

  for idx, subject_dir in enumerate(subject_dirs, 1):
    subject_id = subject_dir.name
    print(f"[{idx}/{total_subjects}] Processing Subject: {subject_id}")

    mri_files = list(subject_dir.glob("mri.nii*"))
    pet_files = list(subject_dir.glob("pet.nii*"))

    if not mri_files or not pet_files:
      print(f"  -> SKIPPING: Missing MRI or PET in {subject_id}")
      continue

    mri_path = str(mri_files[0])
    pet_path = str(pet_files[0])

    mri_out_path = output_path / f"{subject_id}_mri.npy"
    pet_out_path = output_path / f"{subject_id}_pet.npy"

    if mri_out_path.exists() and pet_out_path.exists():
      print("  -> Already processed. Skipping.")
      continue

    try:
      native_mri = ants.image_read(mri_path)
      native_pet = ants.image_read(pet_path)

      print("  -> Aligning PET to MRI...")
      pet_to_mri = ants.registration(
        fixed=native_mri,
        moving=native_pet,
        type_of_transform='Rigid'
      )

      print("  -> Warping MRI to MNI Atlas (This takes time)...")
      mri_to_template = ants.registration(
        fixed=template,
        moving=native_mri,
        type_of_transform='SyN'
      )

      print("  -> Warping PET to MNI Atlas...")
      transforms = mri_to_template['fwdtransforms'] + pet_to_mri[
        'fwdtransforms']
      warped_pet = ants.apply_transforms(
        fixed=template,
        moving=native_pet,
        transformlist=transforms
      )
      warped_mri = mri_to_template['warpedmovout']

      final_mri = ants.resample_image(warped_mri, output_shape, use_voxels=True,
                                      interp_type=0)
      final_pet = ants.resample_image(warped_pet, output_shape, use_voxels=True,
                                      interp_type=0)

      mri_array = final_mri.numpy().astype(np.float32)
      pet_array = final_pet.numpy().astype(np.float32)

      mri_array = (mri_array - np.min(mri_array)) / (
            np.max(mri_array) - np.min(mri_array) + 1e-8)
      pet_array = (pet_array - np.min(pet_array)) / (
            np.max(pet_array) - np.min(pet_array) + 1e-8)

      np.save(mri_out_path, mri_array)
      np.save(pet_out_path, pet_array)
      print("  -> Successfully saved tensors.")

    except Exception as e:
      print(f"  -> ERROR processing {subject_id}: {e}")

  print("\nBatch Harmonization Complete!")
  print(f"All CNN-ready tensors are saved in: {output_path.absolute()}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Batch harmonize ADNI MRI and PET scans.")
  parser.add_argument("input_dir", type=str,
                      help="Path to the flat dataset folder (e.g., ./ADNI_Clean_ML_Ready)")
  parser.add_argument("output_dir", type=str,
                      help="Path to save the final .npy tensors (e.g., ./ADNI_Tensors)")

  args = parser.parse_args()
  batch_harmonize(args.input_dir, args.output_dir)