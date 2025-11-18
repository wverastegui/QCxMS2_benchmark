import re
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AddHs
from rdkit.Chem import AllChem
from typing import Tuple

# ==============================
# Utility atom-checking functions
# ==============================

def has_halogen_atoms(mol: Chem.Mol) -> bool:
    """
    Check if a molecule contains any halogen atoms (F, Cl, Br, I).

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.

    Returns
    -------
    bool
        True if any halogen atoms are present, False otherwise.
    """
    return any(atom.GetSymbol() in ['F', 'Cl', 'Br', 'I'] for atom in mol.GetAtoms())


def has_atom(mol: Chem.Mol, atom: str) -> bool:
    """
    Check if a molecule contains a specific type of atom.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.
    atom : str
        Atom symbol to check for (e.g., 'Cl', 'N').

    Returns
    -------
    bool
        True if the molecule contains the specified atom, False otherwise.
    """
    return any(a.GetSymbol() == atom for a in mol.GetAtoms())


def has_organic_atoms(mol: Chem.Mol) -> bool:
    """
    Check if a molecule contains any organic atoms (C, O, H).

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.

    Returns
    -------
    bool
        True if any C, O, or H atoms are present, False otherwise.
    """
    return any(a.GetSymbol() in ['C', 'O', 'H'] for a in mol.GetAtoms())

# ==============================
# Core workflow functions
# ==============================

def select_best_isomers(df: pd.DataFrame, score_col: str, match_col: str, verbose: bool = False
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select a single representative isomer per query molecule.

    Preference:
        1. Isomer with both highest score and highest matches.
        2. Otherwise, isomer with highest score; tie-break with matches.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing multiple isomers per query.
    score_col : str
        Column name for similarity/quality score.
    match_col : str
        Column name for number of matches or hits.
    verbose : bool, optional
        If True, prints selection summary (default False).

    Returns
    -------
    df_unique : pd.DataFrame
        DataFrame with unique selected isomers.
    both_max_df : pd.DataFrame
        Subset of df_unique where both score and matches were maximal.
    alternative_selection_df : pd.DataFrame
        Subset selected by fallback rule (score max, tie-break by matches).
    """
    selected_rows, both_max_rows, alternative_selection_rows = [], [], []
    df = df.dropna(subset=[score_col, match_col])

    for query, group in df.groupby("query", sort=False):
        if group.empty:
            continue

        max_score = group[score_col].max()
        max_matches = group[match_col].max()
        both_max = group[(group[score_col] == max_score) & (group[match_col] == max_matches)]
        if not both_max.empty:
            row = both_max.sort_values(by=["reference"]).iloc[0]
            both_max_rows.append(row)
        else:
            alt_group = group[group[score_col] == max_score]
            row = alt_group.loc[alt_group[match_col].idxmax()]
            alternative_selection_rows.append(row)
        selected_rows.append(row)

    df_unique = pd.DataFrame(selected_rows).reset_index(drop=True)
    both_max_df = pd.DataFrame(both_max_rows).reset_index(drop=True)
    alternative_selection_df = pd.DataFrame(alternative_selection_rows).reset_index(drop=True)

    if verbose:
        print(f"Selected {len(df_unique)} unique isomers | {len(both_max_df)} both-max | {len(alternative_selection_df)} fallback")
    return df_unique, both_max_df, alternative_selection_df


def categorize_by_score(df: pd.DataFrame, score_col: str, bins: list, labels: list,
                        include_lowest: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Categorize molecules into performance groups based on score intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with score column.
    score_col : str
        Column containing the score values.
    bins : list of float
        Bin edges for categorization.
    labels : list of str
        Labels corresponding to bins.
    include_lowest : bool, optional
        Include the lowest value in the first bin (default True).
    verbose : bool, optional
        Print counts per category (default False).

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'Category' column.
    """
    df["category"] = pd.cut(df[score_col], bins=bins, labels=labels, include_lowest=include_lowest)
    if verbose:
        print("Categorized scores:")
        print(df["category"].value_counts())
    return df


def filter_by_category(df: pd.DataFrame, category: str, verbose: bool = False) -> pd.DataFrame:
    """
    Filter DataFrame for molecules in a specific category.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Category' column.
    category : str
        Category label to filter for.
    verbose : bool, optional
        Print number of molecules selected (default False).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    filtered_df = df[df["category"] == category].reset_index(drop=True)
    if verbose:
        print(f"Filtered {len(filtered_df)} molecules in category '{category}'")
    return filtered_df


def merge_descriptors(df_existing: pd.DataFrame, df_descriptors: pd.DataFrame,
                      left_on: str = "molname", right_on: str = "molname", verbose: bool = False) -> pd.DataFrame:
    """
    Merge molecular descriptors into an existing DataFrame.

    Parameters
    ----------
    df_existing : pd.DataFrame
        DataFrame with query results or filtered TSV.
    df_descriptors : pd.DataFrame
        Descriptor DataFrame extracted from SDF.
    left_on : str
        Column name in df_existing to merge on.
    right_on : str
        Column name in df_descriptors to merge on.
    verbose : bool, optional
        Print merged dataset size (default False).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing descriptors.
    """
    merged_df = pd.merge(df_existing, df_descriptors, left_on=left_on, right_on=right_on, how='left')
    if verbose:
        print(f"Merged descriptors — final dataset contains {len(merged_df)} rows")
    return merged_df


def remove_isotopically_labeled(df: pd.DataFrame, column: str = "query", verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect and remove isotopically labeled compounds from a DataFrame.

    Isotopic labels typically appear as '_2H', '_13C', '_15N', '_18O', etc.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with molecular identifiers.
    column : str
        Column containing molecule names (default 'query').
    verbose : bool, optional
        Print removed molecules info (default False).

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame without isotopically labeled molecules.
    df_removed : pd.DataFrame
        DataFrame containing only removed isotopically labeled molecules.
    """
    isotope_pattern = re.compile(r"_(\d+[A-Z][a-z]?)")
    mask = df[column].astype(str).apply(lambda x: bool(isotope_pattern.search(x)))
    df_removed = df[mask].copy()
    df_clean = df[~mask].copy()

    if verbose:
        print(f"Removed {len(df_removed)} isotopically labeled molecules")
        if not df_removed.empty:
            print(df_removed[column].tolist())
        print(f"Remaining rows after removal: {len(df_clean)}")
    return df_clean, df_removed



def extract_descriptors_from_sdf(sdf_input, verbose: bool = False) -> tuple[pd.DataFrame, list[Chem.Mol]]:
    """
    Extract molecular descriptors from an SDF and retain original RDKit Mol objects.
    """
    molecules = Chem.SDMolSupplier(sdf_input, sanitize=True) if isinstance(sdf_input, str) else sdf_input
    mol_list = [m for m in molecules if m is not None]
    skipped_count = len(molecules) - len(mol_list)
    if verbose and skipped_count:
        print(f"Skipped {skipped_count} molecules due to load errors.")

    rows = []
    for m in mol_list:
        m_h = AddHs(m)
        rows.append({
            "molname": m.GetProp("NAME"),
            "class": m.GetProp("Class"),
            "superclass": m.GetProp("Superclass"),
            "subclass": m.GetProp("Subclass"),
            "n_atoms": m_h.GetNumAtoms(),
            "n_bonds": m_h.GetNumBonds(),
            "inchikey": m.GetProp("InChIKey"),
            "smiles": m.GetProp("SMILES"),
            "has_halogen": has_halogen_atoms(m),
            "Cl": has_atom(m, "Cl"),
            "Br": has_atom(m, "Br"),
            "F": has_atom(m, "F"),
            "S": has_atom(m, "S"),
            "P": has_atom(m, "P"),
            "Si": has_atom(m, "Si"),
            "rotatable_bonds": int(m.GetProp("Rotatable Bonds")),
            "stereo_centers": int(m.GetProp("Stereo Centers")),
            "molecular_complexity": float(m.GetProp("Molecular Complexity")),
            "molecular_flexibility": float(m.GetProp("Molecular Flexibility")),
            "composition": ",".join(sorted({a.GetSymbol() for a in m_h.GetAtoms()}))
        })

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Extracted descriptors for {len(df)} molecules successfully.")
    return df, mol_list


def rename_columns(df: pd.DataFrame, rename_dict: dict, verbose: bool = False) -> pd.DataFrame:
    """
    Rename columns in a DataFrame with optional verbose reporting.

    Args:
        df (pd.DataFrame): Input DataFrame.
        rename_dict (dict): Mapping {old_name: new_name}.
        verbose (bool): Print changes if True (default: False).

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    for old_name, new_name in rename_dict.items():
        if old_name in df.columns and old_name != new_name:
            df.rename(columns={old_name: new_name}, inplace=True)
            if verbose:
                print(f"Renamed column '{old_name}' → '{new_name}'")
    return df



def generate_3D_mol_pristine(
    mol: Chem.Mol,
    max_confs: int = 20,
    rms_cutoff: float = 0.7,
    energy_window: float = 10.0,
    random_seed: int = 42
) -> Chem.Mol:
    """
    Generate a robust, low-energy 3D conformer for a molecule.

    Generates multiple 3D conformers, optimizes them using UFF,
    filters conformers by energy and RMS, and returns a molecule
    containing only the lowest-energy conformer.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        max_confs (int): Maximum number of conformers to generate.
        rms_cutoff (float): Minimum RMSD between conformers for pruning.
        energy_window (float): Maximum energy difference from lowest energy conformer (kcal/mol).
        random_seed (int): Seed for reproducibility.

    Returns:
        rdkit.Chem.Mol: Molecule with a single low-energy conformer.
        None if 3D generation fails.
    """
    if mol is None:
        return None

    try:
        # Work on a copy to preserve the original molecule
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        mol_copy = Chem.AddHs(mol_copy)

        # Embed conformers
        params = AllChem.ETKDGv3()
        params.numThreads = 0
        params.pruneRmsThresh = rms_cutoff
        params.randomSeed = random_seed

        cids = AllChem.EmbedMultipleConfs(mol_copy, numConfs=max_confs, params=params)
        if not cids:
            return None

        # Optimize conformers and compute energies
        energies = []
        for cid in cids:
            AllChem.UFFOptimizeMolecule(mol_copy, confId=cid)
            ff = AllChem.UFFGetMoleculeForceField(mol_copy, confId=cid)
            energies.append(ff.CalcEnergy())

        energies = np.array(energies)
        min_energy = energies.min()
        valid_cids = [cid for cid, e in zip(cids, energies) if e - min_energy <= energy_window]

        if not valid_cids:
            return None

        # RMS pruning
        final_cids = []
        for cid in valid_cids:
            if all(AllChem.GetBestRMS(mol_copy, mol_copy, prev, cid) >= rms_cutoff for prev in final_cids):
                final_cids.append(cid)
            if final_cids:
                break

        best_cid = final_cids[0] if final_cids else valid_cids[0]

        # Create new molecule with single conformer
        new_mol = Chem.Mol(mol_copy)
        conf = mol_copy.GetConformer(best_cid)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf, assignId=True)

        return new_mol

    except Exception as e:
        # Optional:
        return None


def generate_3D_subset(
    mol_list: list,
    df: "pd.DataFrame",
    subset_filter: dict = None
) -> tuple[list, list]:
    """
    Generate 3D conformers for a subset of molecules.

    Args:
        mol_list (list of rdkit.Chem.Mol): Full molecule objects.
        df (pd.DataFrame): DataFrame containing at least 'inchikey'.
        subset_filter (dict, optional): Column-value pairs to select subset.
            Example: {'category': 'very_poor'}.

    Returns:
        tuple:
            - mol_list_3D (list of rdkit.Chem.Mol): Successfully generated 3D molecules.
            - failed_keys (list of str): InChIKeys that failed 3D generation or missing.
    """
    # Subset selection
    if subset_filter:
        mask = True
        for col, val in subset_filter.items():
            mask &= (df[col] == val)
        subset_keys = set(df.loc[mask, "inchikey"])
    else:
        subset_keys = set(df["inchikey"])

    # Map InChIKey -> molecule
    mol_dict = {m.GetProp("InChIKey").strip(): m for m in mol_list}

    mol_list_3D = []
    failed_keys = []

    for ik in subset_keys:
        mol = mol_dict.get(ik)
        if mol:
            mol_3D = generate_3D_mol_pristine(mol)
            if mol_3D:
                mol_list_3D.append(mol_3D)
            else:
                failed_keys.append(ik)
        else:
            failed_keys.append(ik)

    return mol_list_3D, failed_keys


def write_mol_to_xyz(mol: Chem.Mol, filepath: str) -> None:
    """
    Write a single RDKit molecule with a 3D conformer to an XYZ file.

    Format:
      - Line 1: number of atoms
      - Line 2: InChIKey (or 'unknown_inchikey')
      - Lines 3+: atom symbols and XYZ coordinates

    Args:
        mol (Chem.Mol): Molecule with 3D conformer.
        filepath (str): Full path to output XYZ file.
    """
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    try:
        inchikey = mol.GetProp("InChIKey")
    except KeyError:
        inchikey = "unknown_inchikey"

    xyz_lines = [str(n_atoms), inchikey]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz_lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(xyz_lines))


def write_mol_to_xyz(mol: Chem.Mol, filepath: str) -> None:
    """
    Write a single RDKit molecule with a 3D conformer to an XYZ file.

    File format:
        Line 1: Number of atoms
        Line 2: InChIKey (or 'unknown_inchikey' if unavailable)
        Line 3+: Atom symbol and XYZ coordinates (Å)

    Args:
        mol (Chem.Mol): RDKit molecule with a 3D conformer.
        filepath (str): Full path to the output XYZ file.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid molecule: None or empty molecule provided.")

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    inchikey = mol.GetProp("InChIKey") if mol.HasProp("InChIKey") else "unknown_inchikey"

    xyz_lines = [str(n_atoms), inchikey]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz_lines.append(f"{atom.GetSymbol():<2} {pos.x: .6f} {pos.y: .6f} {pos.z: .6f}")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(xyz_lines))


def write_multiple_mols_to_xyz(mol_list: list[Chem.Mol], subfolder: str = "default_set") -> None:
    """
    Write multiple RDKit molecules with 3D conformers to XYZ files
    in a subfolder within 'QCxMS2_benchmark/data/xyz_files'.

    The path is resolved relative to the script location:
        QCxMS2_benchmark/data/scripts/your_script.py
        → QCxMS2_benchmark/data/xyz_files/<subfolder>/

    Args:
        mol_list (list[Chem.Mol]): List of RDKit molecules with 3D conformers.
        subfolder (str): Name of the subfolder inside 'xyz_files' where files will be stored.
                         Defaults to 'default_set'.
    """
    if not mol_list:
        raise ValueError("No molecules provided.")

    # Get absolute path to 'data' directory (one level up from 'scripts')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, ".."))

    # Define xyz_files/<subfolder> directory
    xyz_dir = os.path.join(data_dir, "xyz_files", subfolder)
    os.makedirs(xyz_dir, exist_ok=True)

    for mol in mol_list:
        if not mol.HasProp("NAME"):
            raise KeyError("Each molecule must have a 'NAME' property.")

        molname = mol.GetProp("NAME").strip()
        molname_clean = re.sub(r"[^A-Za-z0-9_\-]+", "_", molname)
        filepath = os.path.join(xyz_dir, f"{molname_clean}.xyz")

        write_mol_to_xyz(mol, filepath)

