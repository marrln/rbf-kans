#!/usr/bin/env python3
"""
Delete specific test version folders from the experiment directory.

If run without --version, it prints all available version folders and exits.
Otherwise, it deletes the specified version (with optional hash filter).

Usage:
    ./delete_test_versions.py --dataset <dataset> [--version <version>] [--hash <hash>] [--dry-run] [--force]
"""

import os
import sys
import shutil
from argparse import ArgumentParser
from collections import defaultdict

# ----------------------------------------------------------------------
# Path setup: same as in the original comparison script
# ----------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

if __name__ == '__main__':
    parser = ArgumentParser(
        description='List or delete specific test version folders (e.g., test_0, final).'
    )
    parser.add_argument(
        '--dataset',
        dest='dataset',
        type=str,
        required=True,
        help='Dataset name (used to locate DATASET_DIR).'
    )
    parser.add_argument(
        '-d', '--test-dir',
        dest='test_dir',
        default=None,
        help='Top directory containing experiment subfolders. Defaults to <dataset>/train.'
    )
    parser.add_argument(
        '--version',
        dest='version',
        default=None,
        help='Version folder name to delete (e.g., "test_0", "final"). If omitted, list all available versions.'
    )
    parser.add_argument(
        '--hash',
        dest='hash',
        default=None,
        help='Optional: only consider versions under this specific configuration hash.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        dest='dry_run',
        help='Print what would be deleted without actually removing anything.'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        dest='force',
        help='Skip the confirmation prompt and delete immediately.'
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Get dataset directory – same as in the original script
    # ------------------------------------------------------------------
    dataset_path = os.path.join(THIS_DIR, args.dataset)
    if dataset_path not in sys.path:
        sys.path.insert(0, dataset_path)
    try:
        from prepare_dataset import DATASET_DIR, DATASET_NAME  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        print(f"Error: Cannot import prepare_dataset from {dataset_path}. Ensure the dataset exists.")
        sys.exit(1)

    if args.test_dir is None:
        args.test_dir = os.path.join(DATASET_DIR, 'train')

    if not os.path.isdir(args.test_dir):
        print(f"Error: Test directory does not exist: {args.test_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Walk the test directory and collect all version folders
    # ------------------------------------------------------------------
    # Structure: test_dir/<hash>/<version>
    # We collect all <version> folders that are direct children of a hash folder.
    version_map = defaultdict(list)  # hash -> list of versions
    for root, dirs, files in os.walk(args.test_dir):
        # Check if current root is a direct child of test_dir (i.e., a hash folder)
        parent = os.path.dirname(root)
        if parent == args.test_dir:
            # root is a hash folder; look for version subfolders
            for d in dirs:
                # We consider any subfolder as a version (no further validation)
                # If hash filter is active, skip hashes that don't match
                if args.hash is not None and os.path.basename(root) != args.hash:
                    continue
                version_map[os.path.basename(root)].append(d)

    if not version_map:
        print(f"No version folders found under '{args.test_dir}'.")
        if args.hash:
            print(f"(Filtered by hash '{args.hash}')")
        sys.exit(0)

    # ------------------------------------------------------------------
    # If no --version was given, print all available versions and exit
    # ------------------------------------------------------------------
    if args.version is None:
        print("Available version folders (grouped by hash):")
        for hash_val, versions in sorted(version_map.items()):
            # Sort and deduplicate versions (though unlikely duplicates per hash)
            unique_versions = sorted(set(versions))
            print(f"  {hash_val}: {', '.join(unique_versions)}")
        # Also print a flat list of all unique version names across all hashes
        all_versions = sorted(set(v for versions in version_map.values() for v in versions))
        print("\nAll unique version names:", ', '.join(all_versions))
        sys.exit(0)

    # ------------------------------------------------------------------
    # Deletion mode: collect folders matching the requested version
    # ------------------------------------------------------------------
    to_delete = []
    for root, dirs, files in os.walk(args.test_dir):
        if os.path.basename(root) == args.version:
            parent_dir = os.path.dirname(root)
            # If hash filter is given, only consider those matching
            if args.hash is not None:
                if os.path.basename(parent_dir) != args.hash:
                    continue
            # Make sure parent is not the test_dir itself (avoids deleting root)
            if parent_dir == args.test_dir:
                # This would mean a version folder directly under test_dir, which is unusual
                # but we can still handle it.
                pass
            to_delete.append(root)

    if not to_delete:
        print(f"No version folders named '{args.version}' found under '{args.test_dir}'.")
        if args.hash:
            print(f"(Filtered by hash '{args.hash}')")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Display what will be deleted and ask for confirmation
    # ------------------------------------------------------------------
    print(f"The following version folders will be deleted ({len(to_delete)} total):")
    for folder in to_delete:
        print(f"  {folder}")

    if args.dry_run:
        print("Dry-run completed. No files were deleted.")
        sys.exit(0)

    if not args.force:
        response = input("Proceed with deletion? [y/N] ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)

    # ------------------------------------------------------------------
    # Perform deletion
    # ------------------------------------------------------------------
    deleted_count = 0
    for folder in to_delete:
        try:
            shutil.rmtree(folder)
            print(f"Deleted: {folder}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {folder}: {e}")

    print(f"Deletion complete. Removed {deleted_count} folder(s).")