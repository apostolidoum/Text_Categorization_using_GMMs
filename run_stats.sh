#!/usr/bin/env bash
# arg1->discard
# arg2->svd_components
# arg3->gmm_components
for discard in 0 500 1000 1500 2000 2500 3000 
do
	for svd_components in 15 20 25 30 35 40 45 50
	do
      	for gmm_components in `seq 1 25`;
      	do
			echo "code.py": discard = $discard svd_components = $svd_components gmm_components=$gmm_components
			python3 "code.py" $discard $svd_components $gmm_components
			echo "<----><---->"
		done
	done 
done
