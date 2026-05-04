import argparse
from pathlib import Path

import numpy as np


def batch_harmonize_resumable(input_dir, output_dir,
    output_shape=(128, 128, 128)):
  input_path = Path(input_dir)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  # 1. Pre-Scan: Determine what is already finished
  all_subject_dirs = [d for d in input_path.iterdir() if d.is_dir()]
  pending_subjects = []
  completed_subjects = []

  for subject_dir in all_subject_dirs:
    subject_id = subject_dir.name
    expected_mri = output_path / f"{subject_id}_mri.npy"
    expected_pet = output_path / f"{subject_id}_pet.npy"

    # A subject is only "completed" if BOTH the MRI and PET tensors exist
    if expected_mri.exists() and expected_pet.exists():
      completed_subjects.append(subject_dir)
    else:
      pending_subjects.append(subject_dir)

  print("--- Resume Status ---")
  print(f"Total Subjects Found: {len(all_subject_dirs)}")
  print(f"Already Completed:    {len(completed_subjects)}")
  print(f"Pending Processing:   {len(pending_subjects)}")
  print("---------------------\n")

  if not pending_subjects:
    print("Dataset is fully harmonized! Nothing left to do.")
    return

  # 2. Only import ANTs if we actually have work to do (saves startup time)
  print("Loading ANTsPy and MNI152 Atlas Template...")
  import ants
  template = ants.image_read(ants.get_ants_data('mni'))

  # 3. Process ONLY the pending subjects
  total_pending = len(pending_subjects)

  for idx, subject_dir in enumerate(pending_subjects, 1):
    subject_id = subject_dir.name
    print(f"[{idx}/{total_pending}] Processing Pending Subject: {subject_id}")

    mri_files = list(subject_dir.glob("mri.nii*"))
    pet_files = list(subject_dir.glob("pet.nii*"))

    if not mri_files or not pet_files:
      print(f"  -> SKIPPING: Missing raw MRI or PET in {subject_id}")
      continue

    mri_path = str(mri_files[0])
    pet_path = str(pet_files[0])

    mri_out_path = output_path / f"{subject_id}_mri.npy"
    pet_out_path = output_path / f"{subject_id}_pet.npy"

    try:
      # Load Images
      native_mri = ants.image_read(mri_path)
      native_pet = ants.image_read(pet_path)

      # Coregister PET to MRI (Rigid)
      print("  -> Aligning PET to MRI...")
      pet_to_mri = ants.registration(fixed=native_mri, moving=native_pet,
                                     type_of_transform='Rigid')

      # Normalize MRI to Template (Non-Linear / SyN)
      print("  -> Warping MRI to MNI Atlas (This takes time)...")
      mri_to_template = ants.registration(fixed=template, moving=native_mri,
                                          type_of_transform='SyN')

      # Apply combined transforms to the PET scan
      print("  -> Warping PET to MNI Atlas...")
      transforms = mri_to_template['fwdtransforms'] + pet_to_mri[
        'fwdtransforms']
      warped_pet = ants.apply_transforms(fixed=template, moving=native_pet,
                                         transformlist=transforms)
      warped_mri = mri_to_template['warpedmovout']

      # Resample to CNN shape
      final_mri = ants.resample_image(warped_mri, output_shape, use_voxels=True,
                                      interp_type=0)
      final_pet = ants.resample_image(warped_pet, output_shape, use_voxels=True,
                                      interp_type=0)

      # Extract and Normalize
      mri_array = final_mri.numpy().astype(np.float32)
      pet_array = final_pet.numpy().astype(np.float32)

      mri_array = (mri_array - np.min(mri_array)) / (
            np.max(mri_array) - np.min(mri_array) + 1e-8)
      pet_array = (pet_array - np.min(pet_array)) / (
            np.max(pet_array) - np.min(pet_array) + 1e-8)

      # Save tensors
      np.save(mri_out_path, mri_array)
      np.save(pet_out_path, pet_array)
      print("  -> Successfully saved tensors.")

    except Exception as e:
      print(f"  -> ERROR processing {subject_id}: {e}")

  print("\nBatch Harmonization Complete!")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Batch harmonize ADNI MRI and PET scans (Resumable).")
  parser.add_argument("input_dir", type=str,
                      help="Path to the flat dataset folder")
  parser.add_argument("output_dir", type=str,
                      help="Path to save the final .npy tensors")

  args = parser.parse_args()
  batch_harmonize_resumable(args.input_dir, args.output_dir)