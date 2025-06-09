for a in 1 2; do
    for nel in 20 40; do
        for o in 2 3 4; do
            for m in 1 $o; do
                for ir in 0 1 2 3; do
                    for pc in 1 2; do
                        filename=meshes/d3_n1_a${a}_nel${nel}_o${o}_m${m}.mesh
                        ./nurbs_lor_solve -patcha -pa -m $filename -int $ir
                    done
                done
            done
        done
    done
done
