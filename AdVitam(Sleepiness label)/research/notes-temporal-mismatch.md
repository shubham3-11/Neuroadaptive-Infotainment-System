# Temporal Mismatch Solutions - Simple Breakdown

**Co-research performed with Claude.ai (Sonnet 4) - Jun 21, 2025**

**Task**: Research and analyze different methodological approaches to handle the temporal mismatch problem between continuous physiological signals (EEG, ECG, EDA) and sparse subjective sleepiness measurements (KSS) in drowsiness detection systems. The goal is to provide evidence-based pros and cons for each approach to guide dataset preparation for the AdVitam physiological drowsiness detection project.

---

## **Approach 1: Linear Interpolation (Current AdVitam Method)**
**What it is**: Fill in missing KSS values by drawing a straight line between known points

**Pros**:
- Simple to implement
- Gives you labels for every segment - "Split each 30-minute session into six 5-minute epochs" and "Assign each epoch a KSS score computed as the linear average of its start- and end-of-segment ratings" ([Meteier et al., 2023](https://www.sciencedirect.com/science/article/pii/S2352340923001452))
- Currently used by AdVitam dataset - "Labeling Simulation: Assign each epoch a KSS score computed as the linear average of its start- and end-of-segment ratings" ([Meteier et al., 2023](https://www.sciencedirect.com/science/article/pii/S2352340923001452))

**Cons**:
- Assumes sleepiness changes linearly - "KSS scores increased 3 hr prior to performance impairment (p < .001) and 4-6 hr prior to physiological sleepiness (p < .001)" and "maximal correlations at positive lags, suggesting KSS was associated with future objective impairment" showing non-linear temporal patterns ([Manousakis et al., 2021](https://pubmed.ncbi.nlm.nih.gov/34032305/))
- Creates artificial data points where no actual measurements exist - Linear interpolation generates labels for time periods when no KSS was actually collected
- Ignores individual variation - "The KSS grade is discrete and poor in temporal resolution" ([Wu et al., 2023](https://www.sciencedirect.com/science/article/abs/pii/S0010482523010557))

---

## **Approach 2: Conservative Labeling**
**What it is**: Only label segments that are very close in time to actual KSS measurements

**Pros**:
- High confidence in labels - "The interrater reliability among sleep technicians has been quantified using Cohen's kappa coefficient in meta-analysis study which resulted in an overall value of 0.76" showing even expert annotations have uncertainty, making conservative labeling prudent ([Phan & Mikkelsen, 2024](https://link.springer.com/article/10.1007/s10462-024-10926-9))
- Avoids interpolation artifacts - Only uses actual measured KSS values rather than generating synthetic labels
- Reduces overfitting to unreliable data - "Self-perceived levels of sleepiness did not correspond reliably to the objectively measured levels" ([Putilov & Donskaya, 2013](https://www.sciencedirect.com/science/article/abs/pii/S1388245713000618))

**Cons**:
- Fewer labeled examples to train on
- Might miss some drowsy periods - "KSS scores increased 3 hr prior to performance impairment" showing sleepiness develops gradually over time periods that conservative labeling might exclude ([Manousakis et al., 2021](https://pubmed.ncbi.nlm.nih.gov/34032305/))
- More complex dataset structure

---

## **Approach 3: Forward Fill/Last Value Carried Forward**
**What it is**: Use the last known KSS value until you get a new one

**Pros**:
- Simple to implement
- More realistic than linear changes - "KSS scores increased 3 hr prior to performance impairment (p < .001) and 4-6 hr prior to physiological sleepiness" showing KSS values can persist over extended periods ([Manousakis et al., 2021](https://pubmed.ncbi.nlm.nih.gov/34032305/))
- Commonly used in medical time series analysis for handling missing data

**Cons**:
- Assumes sleepiness stays constant - "Maximal correlations at positive lags, suggesting KSS was associated with future objective impairment" indicating KSS values do change over time ([Manousakis et al., 2021](https://pubmed.ncbi.nlm.nih.gov/34032305/))
- Still creates labels for unmeasured time periods - Forward fill assigns the same KSS value to segments where no actual measurement was taken
- Can miss gradual changes - "Drowsiness gradually and smoothly accumulates within a short period" ([Wu et al., 2023](https://www.sciencedirect.com/science/article/abs/pii/S0010482523010557))

---

## **Approach 4: Regression Instead of Classification**
**What it is**: Predict drowsiness as a continuous number instead of discrete categories

**Pros**:
- Works better with sparse labels - "Binary classification accuracy for the regression model was 82.6% as compared to 82.0% for a model that was trained specifically for the binary classification task" (Åkerstedt et al., 2021)
- More natural for KSS scale - "The implicit order of the KSS ratings, i.e. the progression from alert to sleepy, provides important information for robust modelling" (Åkerstedt et al., 2021)
- Handles continuous changes - "Drowsiness gradually and smoothly accumulates within a short period" ([Wu et al., 2023](https://www.sciencedirect.com/science/article/abs/pii/S0010482523010557))

**Cons**:
- Harder to interpret results
- Still need reliable ground truth - "Objective sleep metrics only accounted for about one fifth of subjective ratings of sleep" ([Chow et al., 2024](https://www.nature.com/articles/s41598-024-56668-0))
- Different evaluation metrics - Need to validate against objective measures rather than simple accuracy

---

## **Approach 5: Event-Based Detection**
**What it is**: Look for physiological "events" (sudden changes) and match those to KSS measurements

**Pros**:
- Avoids temporal interpolation - "A smoothing process was not required; this is an advantage of the transition-constrained DHMM, which already considers the relationship between sleep stages in transitions" ([Pan et al., 2012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3462123/))
- Focuses on actual physiological changes - "Movement-related components can be traced back to specific events like movement termination and initiation" using temporal alignment ([Miller et al., 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4292555/))
- Natural temporal dependencies - Hidden Markov models "already considers the relationship between sleep stages in transitions" ([Pan et al., 2012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3462123/))

**Cons**:
- Much more complex to implement
- Requires good change detection - "Detection nonstationarity is based on parameter fluctuations" requiring sophisticated algorithms ([Basseville & Nikiforov, 1995](https://www.sciencedirect.com/science/article/abs/pii/002002559500021G))
- May miss gradual onset - "Drowsiness gradually and smoothly accumulates" may not create clear events ([Wu et al., 2023](https://www.sciencedirect.com/science/article/abs/pii/S0010482523010557))

---

## **Approach 6: Avoid KSS Entirely - Use Video/Behavior**
**What it is**: Use video recordings of drowsy behavior instead of subjective ratings

**Pros**:
- Continuous objective data - "Video recording during data collection served as a measure of ground truth in this study" ([Awais et al., 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5620623/))
- No temporal mismatch problem - "Video data was continuously recorded for the whole MD session, and was synchronized with the physiological data acquisition device" ([Awais et al., 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5620623/))
- More reliable than self-report - "These methods do not rely on the driver's self-assessment report and are considered more reliable than subjective methods" ([Khan et al., 2023](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1153268/full))

**Cons**:
- Need video data - "The subjects' faces and physical responses were recorded with multiple cameras from different view angles" requiring extensive setup ([Khan et al., 2023](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1153268/full))
- Subjective annotation of videos - "Data prior to drowsiness-related events were considered to represent active- or alert-state data" still requires human judgment ([Awais et al., 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5620623/))
- Different from KSS-based validation - May not correlate with your target measure

---

## **Approach 7: Multi-Level Temporal Context**
**What it is**: Use multiple time windows (short and long) to capture different temporal patterns

**Pros**:
- Captures temporal dependencies - "Considering the latest two-minute raw single-channel EEG can be a reasonable choice for sleep scoring via deep neural networks with efficiency and reliability" ([Seo et al., 2020](https://www.sciencedirect.com/science/article/pii/S1746809420301932))
- Better performance - "The model achieving the best accuracy when L=10" epochs, with "best accuracy of 83.9% on Sleep-EDF and 86.7% on SHHS datasets" ([Lv et al., 2022](https://onlinelibrary.wiley.com/doi/10.1155/2022/6104736))
- Handles multiple timescales - "Inputs shorter than 17.5 min may reduce performance by restricting the model from observing long-range dependencies" ([Perslev et al., 2021](https://www.nature.com/articles/s41746-021-00440-5))

**Cons**:
- Complex implementation - Requires sophisticated neural architectures
- Still needs labels - "Bidirectional-Long Short-Term Memory to learn transition rules among sleep stages automatically" but still requires ground truth ([Supratak et al., 2017](https://arxiv.org/abs/1703.04046))
- Computational overhead - Processing multiple temporal scales increases complexity

---

## **My Recommendation**

Start with **Approach 2 (Conservative Labeling)** + **Approach 4 (Regression)** because:
1. Conservative labeling addresses reliability issues: "Self-perceived levels of sleepiness did not correspond reliably to the objectively measured levels" ([Putilov & Donskaya, 2013](https://www.sciencedirect.com/science/article/abs/pii/S1388245713000618))
2. Regression performed better: "Binary classification accuracy for the regression model was 82.6% as compared to 82.0%" (Åkerstedt et al., 2021)
3. You can add **Approach 7 (Multi-Level Temporal Context)** later for the 2-minute context windows that research shows are optimal

## **Approach 8: Temporal Shift Optimization**
**What it is**: Automatically find the best time delay between physiological signals and KSS measurements to maximize their alignment

**Pros**:
- Accounts for natural delays in subjective reporting - "Trigeorgis et al. estimated the reaction time as a fixed value (between 0.04 and 10 seconds) that could be found through maximizing the concordance correlation coefficient between real and predicted emotion labels" ([Huang et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC9205566/))
- Systematic approach to finding optimal alignment - "Applied a temporal shift to each training sample to compensate for the annotation delay...The value of the temporal shift, N, is tuned based on the development error during the training procedure" ([Huang et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC9205566/))
- Improves model performance by better matching physiological and subjective data

**Cons**:
- Requires computational search across different time shifts
- May find false optimal shifts if data is noisy
- Adds complexity to the preprocessing pipeline

---

## **Approach 9: Temporal Smoothness Constraints**
**What it is**: Add constraints to machine learning models that force drowsiness predictions to change slowly over time, like real drowsiness does

**Pros**:
- Based on scientific evidence about drowsiness progression - "Emotion states in brain signalling tend to last for about 5–15 s before transitioning to another state" ([Kragel et al., 2022](https://link.springer.com/article/10.1007/s11571-023-10004-w))
- Reduces noisy predictions - "We propose to apply a low-pass filter...to compensate for reaction lag and also remove the unsatisfactory high-frequency components from the output" ([Parthasarathy & Busso, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC9205566/))
- Successful in emotion recognition - "Based on prior knowledge that emotion varies slowly across time, we propose a temporal-difference minimizing neural network (TDMNN)" ([Chen et al., 2024](https://link.springer.com/article/10.1007/s11571-023-10004-w))

**Cons**:
- May miss rapid drowsiness changes (microsleeps)
- Requires tuning of smoothness parameters
- Could over-smooth important transitions

---

## **Approach 10: Multi-Scale Temporal Windows**
**What it is**: Extract features from physiological signals using multiple time windows (short, medium, long) to capture different patterns of drowsiness

**Pros**:
- Captures patterns at different time scales - Video analysis shows "using multiple past frames with intervals to predict future frames" using "four consecutive frames with intervals of one" improves prediction ([Wu et al., 2021](https://www.sciencedirect.com/science/article/abs/pii/S0952197623012411))
- Better temporal coverage - "Signal fragments...long 112 s, and step equals 1 s" creating overlapping contexts ([Ferdinando et al., 2024](https://link.springer.com/article/10.1007/s10994-023-06509-4))
- Proven effective in stress detection using overlapping windows

**Cons**:
- Increases computational complexity
- More features to manage and tune
- Risk of information redundancy across scales

---

## **Approach 11: Onset/Offset Detection**
**What it is**: Instead of labeling every time segment, detect the precise moments when drowsiness starts and ends, then label periods between these events

**Pros**:
- High precision timing - "Manual temporal action localisation aimed to detect a single behavioural pattern using original software with 1 frame precision. For every rear, the frames containing its onset and offset were determined" ([Kurek et al., 2025](https://www.nature.com/articles/s41598-025-89687-6))
- Important for correlating with other signals - "Determination of the precise temporal boundaries of behavioural patterns is of utmost importance when searching for correlations with neural activity" ([Kurek et al., 2025](https://www.nature.com/articles/s41598-025-89687-6))
- Natural way humans experience drowsiness (gradual onset, clear episodes)

**Cons**:
- Requires sophisticated change detection algorithms
- May miss gradual drowsiness that doesn't have clear boundaries
- More complex than simple time-window approaches

---

## **Approach 12: Physiological Window Optimization**
**What it is**: Use research from other physiological monitoring fields to determine optimal time windows for analyzing drowsiness signals

**Pros**:
- Evidence-based window sizing - Pain research shows "experiments are carried out on windows of length 4.5 s with a temporal shift of 4 s from the elicitations' onset" works well ([Werner et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC11004333/))
- Temporal information importance - "Temporal information is very important for pain perception modeling and can significantly increase the prediction accuracy" ([Rish et al., 2010](https://link.springer.com/chapter/10.1007/978-3-642-15314-3_20))
- Similar physiological responses - Pain and drowsiness both involve autonomic nervous system changes

**Cons**:
- Pain and drowsiness may have different optimal time scales
- Research from other domains may not directly transfer
- Still requires validation on drowsiness-specific data

---

## **Cross-Domain Insights Summary**

**Key Finding from Related Research**: Multiple research domains face similar temporal mismatch problems and have developed sophisticated solutions:

**Stress Detection**: Uses overlapping temporal windows and state-based labeling rather than continuous interpolation ([Ferdinando et al., 2024](https://link.springer.com/article/10.1007/s10994-023-06509-4))

**Emotion Recognition**: Developed delay compensation techniques and temporal smoothness constraints that significantly improve performance ([Huang et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC9205566/))

**Video Behavior Analysis**: Achieves frame-level precision for detecting behavior onsets and uses multi-scale temporal analysis ([Kurek et al., 2025](https://www.nature.com/articles/s41598-025-89687-6))

**Pain Assessment**: Shows temporal information dramatically improves physiological-based prediction accuracy ([Rish et al., 2010](https://link.springer.com/chapter/10.1007/978-3-642-15314-3_20))

## **Updated Recommendations**

Start with **Approach 2 (Conservative Labeling)** + **Approach 8 (Temporal Shift Optimization)** + **Approach 9 (Temporal Smoothness Constraints)** because:

1. Conservative labeling provides high-quality training data
2. Temporal shift optimization finds the best alignment between KSS and physiological signals 
3. Smoothness constraints ensure biologically plausible drowsiness progression
4. This combination addresses both data quality and temporal modeling challenges

## **Implementation Priority**

1. **Phase 1**: Conservative labeling with temporal shift optimization (±2.5 minutes around KSS, test different delay offsets)
2. **Phase 2**: Add temporal smoothness constraints to regression models
3. **Phase 3**: Implement multi-scale temporal windows for feature extraction
4. **Phase 4**: Compare with onset/offset detection approaches
5. **Phase 5**: Validate all approaches with subject-independent cross-validation