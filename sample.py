import cv2
import dataclasses
import logging
import numpy as np
import struct

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclasses.dataclass
class Vertex:
    xyz: list[float]
    normal: list[float]
    uv: list[float]
    option_uv: list[float]

    weight_type: int
    weight: bytes
    edge_magnification: float

@dataclasses.dataclass
class Material:
    diffuse: list[float]
    specular: list[float]
    specular_power: float
    ambient: list[float]
    flag: int
    edge_color: list[float]
    edge_size: float
    texture_index: int
    sphere_texture_index: int
    sphere_mode: int
    toon_sharing: int
    toon_texture: int
    memo: str
    num_face: int

verteces = []
materials = []

def load_pmx(file_path):
    with open("models/" + file_path, 'rb') as file:
        # PMXヘッダ
        magic_number = file.read(4)
        logger.info(f"{magic_number=}")
        version: float = struct.unpack("f", file.read(4))[0]
        logger.info(f"{version=}")
        data_byte = file.read(1)
        data_size = int.from_bytes(data_byte, "little")
        logger.info(f"{data_size=}")
        data = file.read(data_size)
        logger.info(f"{data=}")

        if data[0] == 0x00:
            encode = lambda x: x.decode("utf-16")
        else:
            raise
        option_UV_size = data[1]
        vertex_index_size = data[2]
        texture_index_size = data[3]
        material_index_size = data[4]
        bone_index_size = data[5]

        # モデル情報
        size_model_name = int.from_bytes(file.read(4), "little")
        logger.info(f"{size_model_name=}")
        model_name = encode(file.read(size_model_name))
        logger.info(f"{model_name=}")

        size_model_name_en = int.from_bytes(file.read(4), "little")
        logger.info(f"{size_model_name_en=}")
        model_name_en = encode(file.read(size_model_name_en))
        logger.info(f"{model_name_en=}")

        size_comment = int.from_bytes(file.read(4), "little")
        logger.info(f"{size_comment=}")
        comment = encode(file.read(size_comment))
        logger.info(f"{comment=}")

        size_comment_en = int.from_bytes(file.read(4), "little")
        logger.info(f"{size_comment_en=}")
        comment_en = encode(file.read(size_comment_en))
        logger.info(f"{comment_en=}")

        # 頂点情報
        vertex_count = int.from_bytes(file.read(4), "little")
        logger.info(f"{vertex_count=}")
        for i in range(vertex_count):
            vertex = Vertex(
                xyz=[struct.unpack("f", file.read(4))[0] for _ in range(3)],
                normal=[struct.unpack("f", file.read(4))[0] for _ in range(3)],
                uv=[struct.unpack("f", file.read(4))[0] for _ in range(2)],
                option_uv=[struct.unpack("f", file.read(4))[0] for _ in range(option_UV_size)],
                weight_type=int.from_bytes(file.read(1), "little"),
                weight=[],
                edge_magnification=[],
            )
            if vertex.weight_type == 0:
                vertex.weight = [
                    int.from_bytes(file.read(bone_index_size), "little"),
                ]
            elif vertex.weight_type == 1:
                vertex.weight = [
                    int.from_bytes(file.read(bone_index_size), "little"),
                    int.from_bytes(file.read(bone_index_size), "little"),
                    struct.unpack("f", file.read(4))[0],
                ]
            elif vertex.weight_type == 2:
                vertex.weight = [
                    int.from_bytes(file.read(bone_index_size), "little"),
                    int.from_bytes(file.read(bone_index_size), "little"),
                    int.from_bytes(file.read(bone_index_size), "little"),
                    int.from_bytes(file.read(bone_index_size), "little"),
                    struct.unpack("f", file.read(4))[0],
                    struct.unpack("f", file.read(4))[0],
                    struct.unpack("f", file.read(4))[0],
                    struct.unpack("f", file.read(4))[0],
                ]
            else:
                logger.info(f"{vertex.weight_type=}")
                raise
            vertex.edge_magnification = struct.unpack("f", file.read(4))[0]
            verteces.append(vertex)

        # 面情報
        faces = [] 
        face_count = int.from_bytes(file.read(4), "little") // 3
        logger.info(f"{face_count=}")
        for i in range(face_count):
            face = [int.from_bytes(file.read(vertex_index_size), "little") for _ in range(3)]
            faces.append(face)

        # テクスチャ情報
        texture_count = int.from_bytes(file.read(4), "little")
        logger.info(f"{texture_count=}")
        for i in range(texture_count):
            size_texture = int.from_bytes(file.read(4), "little")
            logger.info(f"{size_texture=}")
            texture = encode(file.read(size_texture))
            logger.info(f"{texture=}")

        # マテリアル情報
        material_count = int.from_bytes(file.read(4), "little")
        logger.info(f"{material_count=}")
        for i in range(material_count):
            size_material_name = int.from_bytes(file.read(4), "little")
            logger.info(f"{size_material_name=}")
            material_name = encode(file.read(size_material_name))
            logger.info(f"{material_name=}")

            size_material_name_en = int.from_bytes(file.read(4), "little")
            logger.info(f"{size_material_name_en=}")
            material_name_en = encode(file.read(size_material_name_en))
            logger.info(f"{material_name_en=}")

            diffuse = [struct.unpack("f", file.read(4))[0] for _ in range(4)]
            logger.info(f"{diffuse=}")
            specular = [struct.unpack("f", file.read(4))[0] for _ in range(3)]
            logger.info(f"{specular=}")
            specular_power = struct.unpack("f", file.read(4))[0]
            logger.info(f"{specular_power=}")
            ambient = [struct.unpack("f", file.read(4))[0] for _ in range(3)]
            logger.info(f"{ambient=}")
            flag = int.from_bytes(file.read(1), "little")
            logger.info(f"{flag=}")
            edge_color = [struct.unpack("f", file.read(4))[0] for _ in range(4)]
            logger.info(f"{edge_color=}")
            edge_size = struct.unpack("f", file.read(4))[0]
            logger.info(f"{edge_size=}")
            texture_index = int.from_bytes(file.read(texture_index_size), "little")
            logger.info(f"{texture_index=}")
            sphere_texture_index = int.from_bytes(file.read(texture_index_size), "little")
            logger.info(f"{sphere_texture_index=}")
            sphere_mode = int.from_bytes(file.read(1), "little")
            logger.info(f"{sphere_mode=}")
            toon_sharing = int.from_bytes(file.read(1), "little")
            logger.info(f"{toon_sharing=}")
            if toon_sharing == 0:
                toon_texture = int.from_bytes(file.read(texture_index_size), "little")
            else:
                toon_texture = int.from_bytes(file.read(1), "little")
            
            size_memo = int.from_bytes(file.read(4), "little")
            logger.info(f"{size_memo=}")
            memo = encode(file.read(size_memo))
            logger.info(f"{memo=}")
            num_face = int.from_bytes(file.read(4), "little") // 3
            logger.info(f"{num_face=}")
            materials.append(Material(
                diffuse=diffuse,
                specular=specular,
                specular_power=specular_power,
                ambient=ambient,
                flag=flag,
                edge_color=edge_color,
                edge_size=edge_size,
                texture_index=texture_index,
                sphere_texture_index=sphere_texture_index,
                sphere_mode=sphere_mode,
                toon_sharing=toon_sharing,
                toon_texture=toon_texture,
                memo=memo,
                num_face=num_face,
            ))


    # 3D描画
    max_x = max([v.xyz[0] for v in verteces])
    min_x = min([v.xyz[0] for v in verteces])
    max_y = max([v.xyz[1] for v in verteces])
    min_y = min([v.xyz[1] for v in verteces])
    max_z = max([v.xyz[2] for v in verteces])
    min_z = min([v.xyz[2] for v in verteces])
    logger.info(f"{max_x=}, {min_x=}")
    logger.info(f"{max_y=}, {min_y=}")
    logger.info(f"{max_z=}, {min_z=}")

    max_x_from_face = max([max([verteces[face[i]].xyz[0] for i in range(3)]) for face in faces])
    min_x_from_face = min([min([verteces[face[i]].xyz[0] for i in range(3)]) for face in faces])
    max_y_from_face = max([max([verteces[face[i]].xyz[1] for i in range(3)]) for face in faces])
    min_y_from_face = min([min([verteces[face[i]].xyz[1] for i in range(3)]) for face in faces])
    logger.info(f"{max_x_from_face=}, {min_x_from_face=}")
    logger.info(f"{max_y_from_face=}, {min_y_from_face=}")

    img = np.array([[0, 0, 0] for _ in range(512 * 512)], dtype=np.uint8).reshape(512, 512, 3)

    def convert_x(x):
        return int((x - min_x) / (max_x - min_x) * 511)
    def convert_y(y):
        return 511 - int((y - min_y) / (max_y - min_y) * 511)
    def convert_z(z):
        return int((z - min_z) / (max_z - min_z) * 511)

    m_idx_s = 0
    for material in materials:
        m_faces = faces[m_idx_s:m_idx_s + material.num_face]
        m_idx_s += material.num_face
        color = [int(c * 255) for c in material.edge_color[2::-1]]
        if color == [0, 0, 0]:
            continue
        for face in m_faces:
            for i in range(3 - 1):
                x = convert_x(verteces[face[i]].xyz[0])
                y = convert_y(verteces[face[i]].xyz[1])
                z = convert_z(verteces[face[i]].xyz[2])
                next_x = convert_x(verteces[face[(i + 1) % 3]].xyz[0])
                next_y = convert_y(verteces[face[(i + 1) % 3]].xyz[1])
                cv2.line(img, (x, y), (next_x, next_y), color, thickness=1, lineType=cv2.LINE_4)

    cv2.imwrite("./dst/output.png", img)


file_path = input()

if file_path.endswith(".pmx"):
    load_pmx(file_path)
