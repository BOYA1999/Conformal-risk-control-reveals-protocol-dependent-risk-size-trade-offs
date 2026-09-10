# Data and model sources

Access dates below describe the frozen study inputs. Raw molecular tables and third-party source trees are not bundled in this repository.

## B-XAIC

- Source: `https://huggingface.co/datasets/mproszewska/B-XAIC`
- Frozen revision: `c963b9ee34862b4115dccf7941ba55c9bee16ad0`
- License reported by the dataset card: CC BY-SA 4.0
- Accessed for this study: 2026-08-04
- Expected files and SHA-256:
  - `data.csv`: `14853568ECE75E5C5666C7190E6402CE8D24B3DD3DB171460E668C82C06344FC`
  - `explanations.sdf`: `83C86A6366AFF1DB12A3E439CEAB9A4F275508FAF1BEB5E1B8ADB261465E70AA`
- Role: public molecular-property tasks and atom-rationale annotations used in the benchmark and calibration audits.
- Redistribution: omitted from this package; retrieve the exact revision and comply with CC BY-SA 4.0.

## Graph Attribution

- Source: `https://github.com/google-research/graph-attribution`
- Frozen commit: `03e7495379df26a21395b25c6a14d92dc27fc3b0`
- Repository license: Apache License 2.0
- Accessed for this study: 2026-08-04
- Role: the benzene and logic7/logic8/logic10 benchmark families and their rationale masks.
- Redistribution: the upstream source/data copy is omitted. Retrieve the frozen commit and follow its LICENSE/NOTICE plus the upstream terms for any underlying molecule collection.

## Polaris HCLint

- Benchmark artifact: `polaris/adme-fang-hclint-1`
- Dataset artifact: `polaris/adme-fang-1`
- Source page: `https://polarishub.io/benchmarks/polaris/adme-fang-hclint-1`
- Dataset source cited by Polaris: `https://doi.org/10.1021/acs.jcim.3c00160`
- Dataset license reported by Polaris 0.13.0: CC BY 4.0
- Accessed and frozen: 2026-09-01
- Benchmark MD5: `42af137e8e493bd313071c720f26205c`
- Official split: 2,229 train and 575 hidden-test molecules; target `LOG_HLM_CLint`.
- Materialized CSV SHA-256:
  - train: `8f2936a7ccf2d80110dff05bc9bf7ea115b076684e47e3c37d47fa09c4998f44`
  - test: `60d55038d3d95ccb34a7be441ae1fbed7cb588e1d93c5843461f43aad98a1052`
- Retrieval: install `polaris-lib==0.13.0` and run `python scripts/polaris/polaris_hclint.py prepare --out-dir data`.
- Evaluation boundary: the official Polaris evaluator was used; hidden test targets were not directly accessed. HCLint supplies no atom-level rationale label, so this dataset tests predictivity and explainer compatibility, not rationale coverage.

## CheMeleon foundation checkpoint

- Record: `https://doi.org/10.5281/zenodo.15460715` (version v2)
- File: `chemeleon_mp.pt`, 34,859,448 bytes
- SHA-256: `c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651`
- Direct file used: `https://zenodo.org/records/15460715/files/chemeleon_mp.pt`
- Role: Chemprop 2.2.4 foundation initialization for three fixed fine-tuning seeds (42, 123, 2026).
- Redistribution: not bundled. The Zenodo record displayed no license value at package preparation time; users must confirm the record and linked repository terms before reuse.

## Related source audits

The study inspected the following public repositories while selecting the ADME benchmark route; their source trees are not dependencies bundled here:

- Computational-ADME, commit `b00df003de117ce9e5b381afd886095c5f2af2d5`, MIT: `https://github.com/molecularinformatics/Computational-ADME`
- OpenADMET models, commit `55dd001549885f8d8095af14733d61f36fc046c0`, Apache-2.0: `https://github.com/OpenADMET/openadmet-models`
- Optimus Prime, commit `7e095630c33c92b86914ccbcb31459ce3827309f`, Apache-2.0: `https://github.com/OpenADMET/optimus-prime`

These references document route selection and provenance; their licenses do not change the license of this repository and do not authorize redistribution of unrelated upstream assets.

## MoleculeACE and ChEMBL v29

- Code source: `https://github.com/molML/MoleculeACE`
- Frozen commit: `7e6de0bd2968c56589c580f2a397f01c531ede26`
- MoleculeACE publication: `https://doi.org/10.1021/acs.jcim.2c01073`
- MoleculeACE code license: MIT, applying to the upstream code only.
- Benchmark lineage: MoleculeACE benchmark records derived from ChEMBL v29.
- ChEMBL license: CC BY-SA 3.0, as documented by the `https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions` official FAQ.
- Frozen split-manifest SHA-256: `C71E0E2624AFFE872E368131517FA1A565B81520FED25F87CA4C64B53F83174F`.
- Target rule: retain targets with at least 20 deterministic molecule-disjoint exact-single-cut MMP cliff pairs in both calibration and test; 11 of 15 eligible MoleculeACE targets met this rule.
- Pairing boundary: RDKit `rdMMPA.FragmentMol(maxCuts=1)` with the default cut pattern and `maxCutBonds=20`; the transformation mask is an experimental SAR-linked proxy, not causal or complete ground truth.
- Reconstruction: obtain the frozen MoleculeACE commit, verify the source hashes in `results/a_level/a7_moleculeace_mmp/provenance_public.json`, and run the scripts under `scripts/a_level/a7_moleculeace_mmp/`.
- Redistribution: raw benchmark tables, SMILES-bearing partitions, molecule-level predictions and attributions, and checkpoints are not included.

The structure-free A7 aggregate tables in `results/a_level/a7_moleculeace_mmp/` are kept outside the root MIT scope and carry the directory-level CC BY-SA 3.0 notice.

## Pharmaco-Explainer K4_2ar

- Associated article: `https://doi.org/10.1016/j.jocs.2026.102946`.
- Code source: `https://github.com/AdamSulek/pharmaco-explainer`.
- Frozen code commit: `f40599677e52574b86d8e7ccc6dc842471519cbd`.
- Upstream code license: MIT; the upstream copyright/license notice is retained beside the pharmacophore adapter scripts. No upstream source tree is bundled.
- Data and checkpoint source: `https://huggingface.co/datasets/klimczakjakubdev/pharmaco-explainer`.
- Frozen data/checkpoint commit: `fb38e88acd4b486a4ba9df74dccfa701e05d01b1`.
- Dataset-card license: CC BY 4.0.
- Task: K4_2ar Close, using the released 2D GCN checkpoint. The associated article reports that the benchmark was generated by screening Enamine REAL Space.
- Frozen input SHA-256 values: split table `42e27f2ef90e36c636018a300984f1fca09889951483236dba1f17c8dce06db7`; atom-reference table `2b8e6cb4bfdaa8c180695d6df405c8dde8c3c25cdeda52481fb03f0ad45114cb`; checkpoint `1548a2942559b3c4705f95b510cc3dba30d23107e7f8d4ef1a76c04ad23b9c4b`.
- Reference boundary: the released positive-molecule atom vectors are pharmacophore-computed operational references, not experimental, biological, mechanistic, causal, or expert-annotated atom truth.
- Redistribution: raw tables, SMILES, IDs, labels, row-level predictions, selected molecule slices, atom masks/scores, checkpoints, and the third-party source tree are not included.

The structure-free pharmacophore aggregate tables retain CC BY 4.0 attribution and are not covered by the repository root MIT License. The 512-positive slice is compared against all 24,069 eligible unique positive test molecules; the 472 correctly predicted positives remain a nested descriptive subset. The release contains aggregate distribution and scaffold-coverage summaries, but no molecule or scaffold identities. Reconstruction of the exact selected identities uses the published hash salt, eligibility rules, and source tables locally; the identities themselves are not distributed.

All author-trained and third-party weights are omitted. The omission is a release-scope choice and is not attributed to an invented licensing prohibition. See `REPRODUCIBILITY.md` for the precise regeneration and verification boundaries.

## Current release boundary

The current analyses use the pinned B-XAIC, Graph Attribution, Polaris, MoleculeACE, and K4_2ar sources documented above. The pharmacophore audit uses the released K4_2ar tables, checkpoint, fixed slices, and score definitions; the rings-count audit uses the pinned B-XAIC inputs and external score cache. No additional external dataset is bundled here.

Only structure-free aggregate outputs are included. Raw tables, structures, molecule or source-row identifiers, per-molecule predictions and attributions, checkpoints, score caches, and logs remain external. The provenance map records analysis identities and repository-relative output names; it does not replace a future public release tag or DOI.
