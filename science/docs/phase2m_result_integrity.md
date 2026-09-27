# Phase 2M Result Integrity

Gate status: **PASS**

- Frozen config hash matches expected: yes (`d10d13eb82490bcfb2151f7db2114e38d8896810255b3138d17540cddabd5104`)
- Parameter provenance hash matches: yes (`a66873453502b8883586c80575fd43cc8033fb4cbb129cadec2419da67c8094f`)
- Result manifest and sidecar hash match: yes (`c7c3859a3f40ee32d4d59fbb8623ce6ed92aa07ccbac440682d0eaa3790b092c`)
- Run manifests verified: 92
- Raw records verified: 18068
- Corrected M6 grade endpoints and joint-sample grade fields verified: yes
- Prior M6 runs superseded/invalidated: 38
- All records `paper_eligible=true`: yes
- All record provenance/hash fields match the campaign: yes
- Synthetic-debug dependency: no
- MAPPO/DiffQP learning execution: no
- Locked repository file hashes unchanged: yes
- Figure/table manifests bind generated artifacts to the result-manifest hash: yes

The raw directories are application-level immutable: the campaign writer refuses to overwrite a completed campaign and every raw file is hash-bound through its run manifest and the top-level result manifest. Windows file permissions are not represented as a cross-platform immutability guarantee; integrity is enforced by cryptographic verification and fail-closed reuse.

The source tree was not a Git worktree at execution time. This limitation is recorded as `UNVERSIONED:66047dcf639e8155eba2f23b2292353a4debe1cc270ffbaac4878180713f8d5f` together with the code snapshot hash `9860ffe8e1cb2a58a0b388e7e1c67c6199d43ba0142b903f9ee1dec307356755`; no Git commit is invented.
