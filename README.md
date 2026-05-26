# MultiAgent-Value-Diversity

The official code of Beyond Alignment: Value Diversity as a Collective Property in Multicultural Agent Systems.

## Set up

Python 3.12.0.

Install the packages via 
```bash
conda env create -f environment.yml
```
or 
```bash
pip install -r requirements.txt
```

The data used in experiment are in `data` folder: WVS questions, WVS ground truth, and project file used in section7.

The WVS data are adopted from the CultureSPA project: https://github.com/shaoyangxu/CultureSPA

## Metric Implementation

The `evaluate.py` implements all the metrics: Value Alignment, **Pairwise Diversity** and **Structural Diversity**.

And you can run it to evaluate systems, before this, you should choose the target `model` and `output_dir` below the `if __name__ == "__main__":` line.

## Cultural Agent Inference (Section 5)

For systems with GPT, Claude, Gemini, Grok, Llama models as backbones, you can use the `sec5_infer_api.py` to do inference, which will produce the cultural agent outputs in the `wvs_evaluation` folder.

Below is an example (inference for all cultural agents initalized with gpt-5.4):
```bash
for c in AUS BOL BRA CAN CHN DEU ETH GBR IND KEN MEX NGA NLD NZL RUS THA UKR USA ZWE
do
    python sec5_infer_api.py --culture $c --model gpt-5.4
done
```

For Qwen backbones, you can simply use the `sec5_infer_torch.py` by specify the same input parameters (culture/model).

`wvs_evaluation` folder provides all the outputs, and you can just run `evaluate.py` on them to produce all the results in Table 1.

## System Homogenization Analysis (Section 5.1/5.2/5.3)

The analysis code for each subsection is introduced as below:

- The relationship between Alignment and Diversity (Section 5.1): `sec5_fig1_left.py`; with per-question breakdown: `sec5_fig1_right.py`
- Mixed Backbones (Section 5.2): `sec5_fig2.py`
- Cultural Composition and Agent Count (Section 5.3): `sec5_fig3.py`


## Multi-round Social Exposure (Section 6)

`sec6_social_exposure_api.py` and `sec6_social_exposure_torch.py` provide the first-round social exposure for GPT, Claude, Gemini, Grok, Llama Systems, and Qwen Systems respectively.

For example, you can run: 
```bash
for c in BRA CHN MEX NGA NZL
do
    python sec6_social_exposure_api.py --culture $c --model gpt-5.4
done
```
to generate the results of gpt-5.4 in `wvs_evaluation_interaction` folder.

----

After the first-round interaction, you can further run the `sec6_multi_turn_api.sh` and `sec6_multi_turn_torch.sh` to conduct additional rounds of interaction, like:
```bash
bash sec6_multi_turn_api.sh <model> <max_round>
```
(max_round = 5 in the paper)
The outputs here will be stored in wvs_evaluation_interaction_round{round_id}

----

And you can use `sec6_fig4.py` and `sec6_fig5.py` to summarize the results of first-round social exposure and the multi-round.


## Collective Decision-Making (Section 7)

Run the `sec7_vote.py` to do the Decision-Making in WVS-Participatory Budgeting task:
```bash
for c in AUS BOL BRA CAN CHN DEU ETH GBR IND KEN MEX NGA NLD NZL RUS THA UKR USA ZWE; do
  python sec7_vote.py --culture $c --model claude-opus-4.7 --n_runs 20
done
```
Here, we run for all cultural agents. And the `sec7_fig6.py` further pick the lowest- and highest-diversity system for outcome comparision.

# Others

`choose_culture.py` evaluates human diversity across all \(18^5\) cultural combinations. Based on this, we selected BRA, CHN, MEX, NGA, and NZL for the main experiments, as this combination exhibits top diversity.

# Citation

If you find this repo useful, please cite: uploading ....

