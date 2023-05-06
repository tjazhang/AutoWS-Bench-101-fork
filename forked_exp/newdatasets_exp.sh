#running experiment on amazon high card with different labelling points
#SP 23

source activate AutoWS-Bench-101
export PYTHONPATH="/gscratch/efml/tzhang26/AutoWS-Bench-101-fork"

# for lp in 100 300 500;
# do 
#     python fwrench/applications/pipeline.py --dataset banking77 --lf_selector snuba_multiclass -nlp $lp 
#     python fwrench/applications/pipeline.py --dataset dbpedia --lf_selector snuba_multiclass -nlp $lp 
#     python fwrench/applications/pipeline.py --dataset dbpedia-111 --lf_selector snuba_multiclass -nlp $lp 
#     python fwrench/applications/pipeline.py --dataset dbpedia-219 --lf_selector snuba_multiclass -nlp $lp 
#     python fwrench/applications/pipeline.py --dataset massive-lowcard --lf_selector snuba_multiclass -nlp $lp 

#     python fwrench/applications/pipeline.py --dataset amazon31 --lf_selector snuba_multiclass -nlp 100
# done


# python fwrench/applications/pipeline.py --dataset youtube

# python fwrench/applications/pipeline.py --dataset dbpedia --lf_selector snuba_multiclass -nlp  140
# python fwrench/applications/pipeline.py --dataset massive-lowcard --lf_selector snuba_multiclass -nlp 180 
# python fwrench/applications/pipeline.py --dataset banking77 --lf_selector snuba_multiclass -nlp 770 
# python fwrench/applications/pipeline.py --dataset dbpedia-111 --lf_selector snuba_multiclass -nlp 1110
# python fwrench/applications/pipeline.py --dataset dbpedia-219 --lf_selector snuba_multiclass -nlp 2190
# python fwrench/applications/pipeline.py --dataset amazon31 --lf_selector snuba_multiclass -nlp 310


# python fwrench/applications/pipeline.py --dataset youtube --lf_selector snuba -ef tfidf
python fwrench/applications/pipeline.py --dataset youtube --lf_selector snuba -ef tfidf -mf 300
# python fwrench/applications/pipeline.py --dataset yelp --lf_selector snuba -ef tfidf
# python fwrench/applications/pipeline.py --dataset imdb --lf_selector snuba -ef tfidf


# for m in 100 300 500 700;
# do 
#     python fwrench/applications/pipeline.py --dataset youtube --lf_selector snuba -ef tfidf -mf $m
#     python fwrench/applications/pipeline.py --dataset yelp --lf_selector snuba -ef tfidf -mf $m
#     python fwrench/applications/pipeline.py --dataset imdb --lf_selector snuba -ef tfidf -mf $m
# done

# for m in 100 300 500 700;
# do 
#     python fwrench/applications/pipeline.py --dataset youtube --lf_selector supervised -ef tfidf -mf $m
#     python fwrench/applications/pipeline.py --dataset yelp --lf_selector supervised -ef tfidf -mf $m
#     python fwrench/applications/pipeline.py --dataset imdb --lf_selector supervised -ef tfidf -mf $m
# done

# python fwrench/applications/pipeline.py --dataset youtube --lf_selector snuba -ef bow
# python fwrench/applications/pipeline.py --dataset yelp --lf_selector snuba -ef bow
# python fwrench/applications/pipeline.py --dataset imdb --lf_selector snuba -ef bow


# for m in 100 300 500 700;
# do 
#     python fwrench/applications/pipeline.py --dataset youtube --lf_selector snuba -ef bow -mf $m
#     python fwrench/applications/pipeline.py --dataset yelp --lf_selector snuba -ef bow -mf $m
#     python fwrench/applications/pipeline.py --dataset imdb --lf_selector snuba -ef bow -mf $m
# done


# for m in 100 300 500 700;
# do 
#     python fwrench/applications/pipeline.py --dataset youtube --lf_selector supervised -ef bow -mf $m
#     python fwrench/applications/pipeline.py --dataset yelp --lf_selector supervised -ef bow -mf $m
#     python fwrench/applications/pipeline.py --dataset imdb --lf_selector supervised -ef bow -mf $m
# done
