# Changelog

## [2.0.0](https://github.com/omedeiro/nanowire_tdgl/compare/tdgl3d-v1.0.0...tdgl3d-v2.0.0) (2026-09-05)


### ⚠ BREAKING CHANGES

* **tdgl3d:** divide the supercurrent by the grid spacing
* **tdgl3d:** take the Maxwell coefficient from the vacuum, not the layer ([#13](https://github.com/omedeiro/nanowire_tdgl/issues/13))
* **tdgl3d:** an oxide declared with kappa=0.0 now transmits the field instead of blocking it. Pass magnetic_kappa=0.0 to keep the old behaviour.
* **solver:** use a consistent Peierls phase convention in LPSI

### Added

* add vortex entry dynamics animation and time-dependent test ([cb894b9](https://github.com/omedeiro/nanowire_tdgl/commit/cb894b9945f7347551241e54eeba3aefe909531c))
* **analysis:** add flux-expulsion analysis for rings and holes ([c5ed2f0](https://github.com/omedeiro/nanowire_tdgl/commit/c5ed2f09d274f6b5ae57b8e723412d27c138372e))
* **analysis:** add the GL free energy and exact fluxoid diagnostics ([baff189](https://github.com/omedeiro/nanowire_tdgl/commit/baff18966e3f3af9fe6e58286f8d45be287688d2))
* **core:** add GLUnits for SI ↔ Ginzburg-Landau conversion ([b93f7a3](https://github.com/omedeiro/nanowire_tdgl/commit/b93f7a37e44f5d13eeaea5c0bc0092601c39d124))
* **physics:** compare cross-sections against closed-form GL solutions ([25017c2](https://github.com/omedeiro/nanowire_tdgl/commit/25017c223a9f15741dbcf5eac60b9a86de0c4b79))
* **tdgl3d:** example for a 3x3 array of 4 um holes in an S/I/S Nb stack ([6533a2c](https://github.com/omedeiro/nanowire_tdgl/commit/6533a2ca5e6996e1fe291c1292919f2c5edc7e6c))
* **tdgl3d:** field-cool protocol, vortex census and entry GIF for the hole array ([e59d600](https://github.com/omedeiro/nanowire_tdgl/commit/e59d6008f83295f58cc0a85429b0f45fa9ad4ef7))
* **tdgl3d:** split the vortex census by region, refresh the cost table ([0749451](https://github.com/omedeiro/nanowire_tdgl/commit/07494517465a750acc351693047d95c9ac217309))
* **tdgl3d:** stream the history to disk, and run in single precision ([b61e8c6](https://github.com/omedeiro/nanowire_tdgl/commit/b61e8c64caae8e848d1f7db727399daa9e7863e2))
* **tdgl3d:** vacuum padding around a stack, and refuse periodic+applied field ([56ca63c](https://github.com/omedeiro/nanowire_tdgl/commit/56ca63c6c699d9e23824ea0423f032f9de7df304))


### Fixed

* **mesh:** carve centred holes and stack layers symmetrically ([cb19627](https://github.com/omedeiro/nanowire_tdgl/commit/cb196270f0c553ac1bbb4bd2c17bce0caa976b3d))
* **mesh:** correct interior-array strides in bfield_interior and eval_bfield ([126bcc7](https://github.com/omedeiro/nanowire_tdgl/commit/126bcc750db66aea6f3540b3992b8e37b7c21267))
* **solver:** correct dφ/dt curl-curl coupling; add physics validation suite ([1c35eb4](https://github.com/omedeiro/nanowire_tdgl/commit/1c35eb42d85f4448487eac741b2d99e63ad88243))
* **solver:** correct dφ/dt curl-curl operator coupling in RHS ([bfa53d2](https://github.com/omedeiro/nanowire_tdgl/commit/bfa53d2af3c16468d85e06844afb41a98bca2935))
* **solver:** stop double-counting the applied flux at hi/hi boundary corners ([4a9a8ef](https://github.com/omedeiro/nanowire_tdgl/commit/4a9a8ef7f8ba194811206ccf028647e498e2c6a5))
* **solver:** use a consistent Peierls phase convention in LPSI ([fabe9a2](https://github.com/omedeiro/nanowire_tdgl/commit/fabe9a228170535f2ff38f75582a7aaf347b5abb))
* **tdgl3d:** divide the supercurrent by the grid spacing ([0533251](https://github.com/omedeiro/nanowire_tdgl/commit/0533251044e095f6df6bc995001f72d45b82e95c))
* **tdgl3d:** draw node-sampled data on cell edges everywhere, not just the S/I/S figure ([4c30c20](https://github.com/omedeiro/nanowire_tdgl/commit/4c30c20b01954b0fa4c50b1768e6b64b3e1a755a))
* **tdgl3d:** rebuild the cached κ² when κ changes on the same grid ([50d3d54](https://github.com/omedeiro/nanowire_tdgl/commit/50d3d54366ac09028c777e9bf7bdb15468d56847))
* **tdgl3d:** take the Maxwell coefficient from the vacuum, not the layer ([1c5bb57](https://github.com/omedeiro/nanowire_tdgl/commit/1c5bb57921b5c680678b3c636e17492d63bcdbe6))
* **tdgl3d:** take the Maxwell coefficient from the vacuum, not the layer ([#13](https://github.com/omedeiro/nanowire_tdgl/issues/13)) ([a8034d4](https://github.com/omedeiro/nanowire_tdgl/commit/a8034d4c5157d87ab18343841fba841b078abd14))
* **tests:** the CFL limit depends on dimension; h²/(4κ²) is the 2-D case ([5b60522](https://github.com/omedeiro/nanowire_tdgl/commit/5b60522dda06699e53b9f2c6fdb1d2eaedc1d905))


### Changed

* **tdgl3d:** drop the gathers and the boundary copies; make single precision pay ([5e3ec98](https://github.com/omedeiro/nanowire_tdgl/commit/5e3ec986cced1e61096eb95c32ea8cc09de82c1e))
* **tdgl3d:** drop the gathers and the boundary copies; make single precision pay ([39abdcf](https://github.com/omedeiro/nanowire_tdgl/commit/39abdcf1538ea06e03fcdda00fc188168995c725))
* **tdgl3d:** drop the gathers and the boundary copies; make single precision pay ([#16](https://github.com/omedeiro/nanowire_tdgl/issues/16)) ([5e3ec98](https://github.com/omedeiro/nanowire_tdgl/commit/5e3ec986cced1e61096eb95c32ea8cc09de82c1e))
* **tdgl3d:** make device-scale Nb films tractable — 172x, plus flux trapping in a 3x3 hole array ([b3daafc](https://github.com/omedeiro/nanowire_tdgl/commit/b3daafc09b59eb8b0aa0c5117b1f964c91b7855e))
* **tdgl3d:** make large 3-D films tractable — 3.5x on the RHS, 160x on hole carving ([6c05704](https://github.com/omedeiro/nanowire_tdgl/commit/6c05704c2fcec5fc1671da7a621cf1767c287ccf))
* **tdgl3d:** thread the right-hand side and stop reallocating the full grid ([3a6e710](https://github.com/omedeiro/nanowire_tdgl/commit/3a6e71015202b8cde961872cdc725b09d434c37f))
* **tdgl3d:** walk the right-hand-side stencil in grid order ([9696b45](https://github.com/omedeiro/nanowire_tdgl/commit/9696b45228ee2b10dc10f5c42b1aef8c32b765aa))

## [1.0.0](https://github.com/omedeiro/nanowire_tdgl/compare/tdgl3d-v0.1.0...tdgl3d-v1.0.0) (2026-07-25)


### ⚠ BREAKING CHANGES

* restructure into monorepo with packages/tdgl3d

### Documentation

* establish AGENTS.md hierarchy for platform monorepo ([396ba0a](https://github.com/omedeiro/nanowire_tdgl/commit/396ba0a496068bce94c0c36e8612144e04bfdcf4))


### Code Refactoring

* restructure into monorepo with packages/tdgl3d ([61ed8ed](https://github.com/omedeiro/nanowire_tdgl/commit/61ed8ed413dd7e326cec41ea8cf3ff5ae5668bd9))
