#/bin/sh

function=$1

if [ $# -ge 1 ]; then

    if [ $function = 'catchcp' ]; then 
    
        python3 catch_contact_point.py --mat-input-dir 'viewer_out' --pcd-input-dir 'mesh_model' --obj 'tt_scissor' --output-dir 'contact_point'
    
    elif [ $function = 'camcap' ]; then 
    
        if [ $# -eq 2 ]; then
            OUT_DIR=$2
            python camera_capturer.py --root 'cam_output' --out_dir ${OUT_DIR}
        elif [ $# -eq 3 ]; then
            ROOT=$2
            OUT_DIR=$3
            python camera_capturer.py --root ${ROOT} --out_dir ${OUT_DIR}
        else 
            python camera_capturer.py --root 'cam_output' --out_dir '0531-1'
        fi
    
    elif [ $function = 'pcd' ]; then 
    
        if [ $# -eq 2 ]; then
            IN_DIR=$2
            python point_cloud.py --root 'cam_output' --in_dir ${IN_DIR} --out_dir ${IN_DIR}
        elif [ $# -eq 3 ]; then
            ROOT=$2
            IN_DIR=$3
            python point_cloud.py --root ${ROOT} --in_dir ${IN_DIR} --out_dir ${IN_DIR}
        else 
            python point_cloud.py --root 'cam_output' --in_dir '0531-1' --out_dir '0531-1'
        fi
    
    elif [ $function = 'pcdshow' ]; then 

        if [ $# -eq 3 ]; then
            ROOT=$2
            OBJ=$3
            python ply_show.py --root $ROOT --obj $OBJ
        else 
            python ply_show.py --root 'mesh_model' --obj 'scissor'
        fi
    
    else
    
        echo "function error : $function"

    fi 

else

    echo 'should at least pass one function as argument'

fi