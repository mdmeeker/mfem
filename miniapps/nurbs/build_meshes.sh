# quick and dirty script to build meshes for testing LOR preconditioning
rm -rf meshes
mkdir meshes

# fixed: 2d, 1 patch
# a = {1 (uniform), 2}
# nel = {10, 30, 100, 300}
# o = {2, 3, 4}
# m = {1, o}
for a in 1 2; do
    for nel in 10 30 100 300; do
        for o in 2 3 4; do
            for m in 1 $o; do
                ./nurbs_lor_cartesian -d 2 -n 1 -a $a -nel $nel -o $o -m $m
                mv ho_mesh.mesh meshes/2d_n1_a${a}_nel${nel}_o${o}_m${m}.mesh
                rm lo_mesh.mesh
            done
        done
    done
done
