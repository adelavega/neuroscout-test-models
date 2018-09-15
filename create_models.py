from pyns import Neuroscout
api = Neuroscout('delavega@utexas.edu', 'vOrlon24!')

# Explore feeatures
datasets = api.datasets.get()
sherlock_preds = [p for p in api.predictors.get(run_id=datasets[0]['runs']) if p['source'] == 'extracted']
len(sherlock_preds)

life_preds = [p for p in api.predictors.get(run_id=datasets[1]['runs']) if p['source'] == 'extracted']
len(life_preds)

set([f['name'] for f in life_preds]) - set([f['name'] for f in sherlock_preds])

visual_features = ['brightness', 'vibrance', 'num_colors_0.07', 'shot_change',
                   'animal', 'building', 'car', 'city', 'furniture', 'indoors', 'landscape', 'man', 'military', 'nature', 'road', 'sky', 'street', 'technology', 'travel', 'wildlife']

face = ['face_detectionConfidence']

auditory = ['rmse']
language = ['speech', 'subtlexusfrequency_Lg10WF']

noise = ['tCompCor{:02d}'.format(i) for i in range(0, 6)]
cosine = ['Cosine{:02d}'.format(i) for i in range(0, 15)]
confounds = noise + cosine + ['FramewiseDisplacement', 'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ', 'WhiteMatter']


# Build analyses
frequency = api.analyses.create_analysis(
    dataset_name='SherlockMerlin', name='Frequency',
    predictor_names=language + confounds,
    hrf_variables=language,
    transformations = [
        {
          "input": "subtlexusfrequency_Lg10WF",
          "name": "scale",
          "replace_na": "after"
        },
        {
          "other": [
            "speech"
          ],
          "input": "subtlexusfrequency_Lg10WF",
          "name": "orthogonalize"
        }
      ]
)

frequency.hash_id
frequency.compile()
frequency.get_status()
