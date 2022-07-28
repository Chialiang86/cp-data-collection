#/bin/sh

function=$1

if [ $# -ge 1 ]; then

    if [ $function = 'camcap' ]; then 
    
        if [ $# -eq 2 ]; then
            OUT_DIR=$2
            python3 camera_capturer.py --root 'cam_output' --out_dir ${OUT_DIR}
        elif [ $# -eq 3 ]; then
            ROOT=$2
            OUT_DIR=$3
            python3 camera_capturer.py --root ${ROOT} --out_dir ${OUT_DIR}
        else 
            python3 camera_capturer.py --root 'cam_output' --out_dir '0531-1'
        fi
    
    elif [ $function = 'catchcp' ]; then 
        if [ $# -eq 2 ]; then
            INPUT=$2
            python3 catch_contact_point.py --root 'cam_output' --in-dir ${INPUT} --out-dir 'annotation'
        elif [ $# -eq 3 ]; then
            INPUT=$2
            OUTPUT=$3
            python3 catch_contact_point.py --root 'cam_output' --in-dir ${INPUT} --out-dir ${OUTPUT}
        elif [ $# -eq 4 ]; then
            ROOT=$2
            INPUT=$3
            OUTPUT=$4
            python3 catch_contact_point.py --root ${ROOT} --in-dir ${INPUT} --out-dir ${OUTPUT}
        else 
            python3 catch_contact_point.py --root 'cam_output' --in-dir '0531-1' --out-dir 'annotation'
        fi
    
    elif [ $function = 'rgbd2mat' ]; then 
        if [ $# -eq 2 ]; then
            INPUT=$2
            python3 rgbd_to_mat.py --input-root 'hcis-data' --input-dir ${INPUT} --output-root 'cam_output'
        else 
            python3 rgbd_to_mat.py --input-root 'hcis-data' --input-dir '20220727_172155_scissor' --output-root 'cam_output'
        fi

    elif [ $function = 'srecon' ]; then 
        if [ $# -eq 2 ]; then
            INPUT=$2
            python3 scene_reconstruct.py --root 'cam_output' --in-dir ${INPUT} --out-dir ${INPUT}
        else 
            python3 scene_reconstruct.py --root 'cam_output' --in-dir '0531-1' --out-dir 'annotation'
        fi
    
    elif [ $function = 'somatch' ]; then 
        if [ $# -eq 3 ]; then
            SCENE_PCD="3d_scene/$2/pcd.ply"
            SCENE_JSON="3d_scene/$2/pcd.json"
            OBJ_PCD="3d_model/$3/$3_pcd.ply"
            OBJ_JSON="3d_model/$3/$3_pcd.json"
            python3 scene_obj_matching.py --scene_pcd $SCENE_PCD --scene_json $SCENE_JSON --obj_pcd $OBJ_PCD --obj_json $OBJ_JSON
        fi

    elif [ $function = 'pcd' ]; then 
    
        if [ $# -eq 2 ]; then
            IN_DIR=$2
            python3 point_cloud.py --root 'cam_output' --in_dir ${IN_DIR} --out_dir ${IN_DIR}
        elif [ $# -eq 3 ]; then
            ROOT=$2
            IN_DIR=$3
            python3 point_cloud.py --root ${ROOT} --in_dir ${IN_DIR} --out_dir ${IN_DIR}
        else 
            python3 point_cloud.py --root 'cam_output' --in_dir '0531-1' --out_dir '0531-1'
        fi
    
    elif [ $function = 'mesh' ]; then 

        if [ $# -eq 2 ]; then
            INPUT=$2
            python3 mesh.py --root 'cam_output' --in-dir ${INPUT} --out-dir 'annotation'
        elif [ $# -eq 3 ]; then
            INPUT=$2
            OUTPUT=$3
            python3 mesh.py --root 'cam_output' --in-dir ${INPUT} --out-dir ${OUTPUT}
        elif [ $# -eq 4 ]; then
            ROOT=$2
            INPUT=$3
            OUTPUT=$4
            python3 mesh.py --root ${ROOT} --in-dir ${INPUT} --out-dir ${OUTPUT}
        else 
            python3 mesh.py --root 'cam_output' --in-dir '0531-1' --out-dir 'annotation'
        fi
    
    else
    
        echo "function error : $function"

    fi 

else

    echo 'should at least pass one function as argument'

fi