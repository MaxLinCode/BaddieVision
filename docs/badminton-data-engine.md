# Badminton data engine

This project is not just a badminton classifier. It should be treated as a
badminton data engine.

That distinction changes the architecture: feature extractors, weak labelers,
review tools, datasets, and models should be separable pieces. Models consume
the dataset; they are not the center of the system.

## Pipeline

```text
Raw Video
    |
    v
Feature Extraction Layer
----------------------------------------
TrackNet
Pose Estimation
Court Calibration
Player Ground Projection
Optional YOLO
    |
    v
Structured Observations
----------------------------------------
Frame
Shuttle position
Player positions
Velocities
Confidence
Court coordinates
    |
    v
Weak Label Generation
----------------------------------------
State machine
Physics rules
Heuristics
Temporal smoothing
    |
    v
Candidate Events / Labels
----------------------------------------
Rallies
Hits
Shot candidates
Serve candidates
Out-of-play
    |
    v
Human Review Interface
----------------------------------------
Review only:
- uncertain
- inconsistent
- novel
- disagreement
    |
    v
Corrected Dataset
----------------------------------------
Versioned
Searchable
High quality
    |
    v
Model Training
----------------------------------------
Rally model
Shot classifier
Hit detector
Future models
    |
    v
Better predictions
    |
    +--------------------+
                         |
                         v
              Better weak labels
```

## Component goals

### Feature extraction

Convert video into reusable structured observations.

This layer should almost never need to be rerun. Raw model outputs such as
shuttle tracks, poses, court calibration, player ground projection, and future
detectors should be preserved as reusable observations rather than collapsed
directly into one task-specific training array.

### Weak label generation

Produce labels that are good enough.

The weak label generator does not need to be perfect. Its job is to make editing
much faster than labeling from scratch. State machines, physics rules,
heuristics, and temporal smoothing should produce candidate events with
confidence, flags, and provenance.

### Review tool

The review tool is a correction tool, not a labeling tool.

It should answer focused questions in a few clicks:

- Is this rally boundary correct?
- Is this hit correct?
- Is this shot correct?

The review queue should prioritize uncertain, inconsistent, novel, or
disagreeing candidates instead of forcing humans to inspect everything.

### Dataset

The dataset should become a durable asset.

Every reviewed match should increase its value. Corrections should be
versioned, searchable, and traceable back to the raw source and the automatic
candidate that produced them. Never lose corrections.

### Models

Models are consumers of the dataset.

The data engine should remain intact even if the project later replaces
TrackNet, the rally segmenter, the shot classifier, or any future model. Model
training should depend on stable reviewed data contracts, not on fragile
intermediate scripts.

## Long-term feedback loop

```text
Match 1
    |
    v
Pipeline
    |
    v
Human corrections
    |
    v
Better dataset
    |
    v
Better model
    |
    v
Pipeline improves
    |
    v
Match 2
    |
    v
Less correction needed
    |
    v
Better dataset
    |
    v
Better model
```

Human effort should decrease over time. That is the hallmark of a good data
engine.

## Guiding principle

Every processed badminton match should increase the capability of the system
while requiring less human effort than the previous one.

The objective is not maximum accuracy in isolation, and it is not the fanciest
model. The objective is a system that continuously compounds knowledge by
turning raw videos into reusable data, using automation for repetitive work and
reserving human attention for ambiguous, high-value decisions.

## Scaffolding direction

The next architecture increment should be modest and modular:

- Define shared observation and candidate-event records before adding more
  task-specific feature arrays.
- Preserve current working pipelines while adding stable seams around their
  inputs and outputs.
- Store candidate labels, human corrections, and reviewed datasets with
  provenance.
- Keep `src/TrackNetV3/` vendored and replaceable.
- Keep `InPlay/` rally logic separate until shared data contracts are stable.
- Treat shot-classifier features as one model-specific consumer of the dataset,
  not as the canonical project data model.
