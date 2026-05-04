import argparse
import subprocess
from pathlib import Path


def automate_dcm2niix(dataset_dir):
  root_path = Path(dataset_dir)

  if not root_path.exists() or not root_path.is_dir():
    print(
      f"Error: The directory '{dataset_dir}' does not exist or is not a folder.")
    return

  print(f"Scanning '{root_path}' for DICOM files. This may take a moment...")

  dcm_folders = set()
  for dcm_file in root_path.rglob('*.dcm'):
    dcm_folders.add(dcm_file.parent)

  if not dcm_folders:
    print("No DICOM files (.dcm) found in the specified directory.")
    return

  print(f"Found {len(dcm_folders)} scan directories to convert.\n")

  for i, folder in enumerate(dcm_folders, 1):
    print(f"[{i}/{len(dcm_folders)}] Processing: {folder.parts[-1]} ...")

    cmd = [
      './dcm2niix',
      '-z', 'y',  # Compress the output into .nii.gz
      '-ba', 'n',  # Do not generate BIDS sidecar JSON files
      '-f', '%n_%p_%s',
      # Filename format: PatientName_ProtocolName_SeriesNumber
      '-o', str(folder),
      # OUTPUT DIRECTORY: Set to the same folder as the DICOMs
      str(folder)  # INPUT DIRECTORY: The folder containing the DICOMs
    ]

    try:
      subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                     stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
      print(f"  -> ERROR converting {folder}")
      print(f"  -> {e.stderr.decode('utf-8')}")

  print(
    "\nConversion complete! All .nii.gz files have been saved alongside their original .dcm files.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Batch convert ADNI DICOM directories to NIfTI format.")
  parser.add_argument("dataset_dir", type=str,
                      help="Path to the root ADNI dataset directory")

  args = parser.parse_args()

  automate_dcm2niix(args.dataset_dir)
