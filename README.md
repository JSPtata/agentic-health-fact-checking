# agentic-health-fact-checking
# Agentic AI for Claim-Level Health Misinformation Verification

## Problem Statement
Given a short-form health or nutrition video containing one or more factual claims, the system extracts the spoken content, decomposes it into individual claims, retrieves reliable external evidence, and classifies each claim as supported, refuted, misleading, or insufficient evidence.

## Objectives
- Extract claims from short-form health videos
- Decompose complex claims into atomic claims
- Retrieve reliable health-related evidence
- Verify each claim independently
- Compare multi-agent verification with single-agent and whole-video baselines

## Proposed Architecture
Video → Preprocessing → Claim Decomposition Agent → Evidence Retrieval Agent → Verification Agent → Adjudication Agent → Claim-Level Verdict

## Dataset
Planned datasets:
- HealthFC
- Short-form health/nutrition video samples
- Additional short-video benchmark datasets if required

## Team Members
- Member 1: Jaya Sai Pranathi, 24BCE1804
- Member 2: Hari Sri Sai, 24BCE1229

## Current Status
DA1:
- Literature survey completed
- Research gap identified
- Problem statement finalized
- Proposed architecture designed
- Dataset loading and preprocessing under development
