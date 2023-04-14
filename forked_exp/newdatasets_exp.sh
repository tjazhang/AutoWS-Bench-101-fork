#running experiment on amazon high card with different labelling points
#SP 23

source activate AutoWS-Bench-101
export PYTHONPATH="/gscratch/efml/tzhang26/AutoWS-Bench-101-fork"

for lp in 100 300 500;
do 
    python fwrench/applications/pipeline.py --dataset banking77 --lf_selector snuba_multiclass -nlp $lp 
    python fwrench/applications/pipeline.py --dataset dbpedia --lf_selector snuba_multiclass -nlp $lp 
    # python fwrench/applications/pipeline.py --dataset dbpedia-111 --lf_selector snuba_multiclass -nlp $lp 
    python fwrench/applications/pipeline.py --dataset dbpedia-219 --lf_selector snuba_multiclass -nlp $lp 
    # python fwrench/applications/pipeline.py --dataset massive-lowcard --lf_selector snuba_multiclass -nlp $lp 
done
