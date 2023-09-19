# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use faiss to build index
"""
import json
import os
from glob import glob
from typing import Optional, Union, List

import faiss
import fire
import numpy as np
from loguru import logger
from text2vec import SentenceModel

from similarities.utils.util import cos_sim


def bert_embedding(
        input_dir: str,
        embeddings_dir: str = 'tmp_embeddings_dir/',
        embeddings_name: str = 'emb.npy',
        corpus_file: str = 'tmp_data_dir/corpus.npy',
        model_name: str = "shibing624/text2vec-base-chinese",
        batch_size: int = 32,
        device: Optional[str] = None,
):
    sentences = set()
    input_files = glob(f'{input_dir}/**/*.txt', recursive=True)
    logger.info(f'Load input files success. input files: {input_files}')
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.add(line)
    sentences = list(sentences)
    logger.info(f'Load sentences success. sentences num: {len(sentences)}, top3: {sentences[:3]}')
    assert len(sentences) > 0, f"sentences is empty, please check input files: {input_files}"

    model = SentenceModel(model_name_or_path=model_name, device=device)
    logger.info(f'Load model success. model: {model}')

    # Start the multi processes pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    # Compute the embeddings using the multi processes pool
    emb = model.encode_multi_process(sentences, pool, batch_size=batch_size)
    logger.info(f"Embeddings computed. Shape: {emb.shape}")

    model.stop_multi_process_pool(pool)
    # Save the embeddings
    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings_file = os.path.join(embeddings_dir, embeddings_name)
    np.save(embeddings_file, emb)
    logger.debug(f"Embeddings saved to {embeddings_file}")
    corpus_dir = os.path.dirname(corpus_file)
    os.makedirs(corpus_dir, exist_ok=True)
    np.save(corpus_file, sentences)
    logger.debug(f"Sentences saved to {corpus_file}")
    logger.info(f"Input dir: {input_dir}, saved embeddings dir: {embeddings_dir}")


def bert_index(
        embeddings_dir: str,
        index_dir: str = "tmp_index_dir/",
        index_name: str = "faiss.index",
        max_index_memory_usage: str = "4G",
        current_memory_available: str = "8G",
        use_gpu: bool = False,
        nb_cores: Optional[int] = None,
):
    """indexes text embeddings using autofaiss"""
    from autofaiss import build_index  # pylint: disable=import-outside-toplevel

    logger.debug(f"Starting build index from {embeddings_dir}")
    if embeddings_dir and os.path.exists(embeddings_dir):
        logger.debug(
            f"Embedding path exist, building index "
            f"using embeddings {embeddings_dir} ; saving in {index_dir}"
        )
        index_file = os.path.join(index_dir, index_name)
        index_infos_path = os.path.join(index_dir, index_name + ".json")
        try:
            build_index(
                embeddings=embeddings_dir,
                index_path=index_file,
                index_infos_path=index_infos_path,
                max_index_memory_usage=max_index_memory_usage,
                current_memory_available=current_memory_available,
                nb_cores=nb_cores,
                use_gpu=use_gpu,
            )
            logger.info(f"Index {embeddings_dir} done, saved in {index_file}, index infos in {index_infos_path}")
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Index {embeddings_dir} failed, {e}")
            raise e
    else:
        logger.warning(f"Embeddings dir {embeddings_dir} not exist")


def _search_index(
        query,
        model,
        index,
        sentences,
        num_results,
        threshold
):
    """Search index with text input"""
    inputs = query
    if isinstance(query, str):
        inputs = [query]
    # Query embeddings need to be normalized for cosine similarity
    query_features = model.encode(inputs, normalize_embeddings=True)

    if threshold is not None:
        _, d, i = index.range_search(query_features, threshold)
        logger.debug(f"Found {i.shape} items with query '{query}' and threshold {threshold}")
    else:
        d, i = index.search(query_features, num_results)
        logger.debug(f"Found {num_results} items with query '{query}'")
        i = i[0]
        d = d[0]

        min_d = min(d)
        max_d = max(d)
        if max_d - min_d < 20:
            logger.debug(f"The minimum distance is {min_d:.2f} and the maximum is {max_d:.2f}")
            logger.debug(
                "You may want to use these numbers to increase your --num_results parameter. "
                "Or use the --threshold parameter."
            )

    # Sorted faiss search result with distance
    text_scores = []
    for ed, ei in zip(d, i):
        sentence = sentences[ei]
        logger.debug(f"Found: {sentence}, similarity: {ed}, id: {ei}")
        text_scores.append((sentence, float(ed), int(ei)))
    # Sort by score desc
    return sorted(text_scores, key=lambda x: x[1], reverse=True)


def bert_filter(
        query: Union[str, List[str]],
        output_file: str = "tmp_outputs/result.json",
        model_name: str = "shibing624/text2vec-base-chinese",
        index_dir: str = 'tmp_index_dir/',
        index_name: str = "faiss.index",
        corpus_file: str = "tmp_data_dir/corpus.npy",
        num_results: int = 10,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
):
    """Entry point of bert filter"""
    assert isinstance(query, str) or isinstance(query, list), f"query type error, query: {query}"
    index_file = os.path.join(index_dir, index_name)
    assert os.path.exists(index_file), f"index file {index_file} not exist"
    index = faiss.read_index(index_file)
    model = SentenceModel(model_name_or_path=model_name, device=device)
    sentences = np.load(corpus_file)
    logger.info(f'Load model success. model: {model}, index: {index}, sentences size: {len(sentences)}')

    sorted_text_scores = _search_index(query, model, index, sentences, num_results, threshold)
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                {'query': query,
                 'results': [{'sentence': i, 'similarity': j, 'id': k} for i, j, k in sorted_text_scores]},
                f,
                ensure_ascii=False,
                indent=2
            )
        logger.info(f"Query: {query}, saved result to {output_file}")
    return sorted_text_scores


def bert_server(
        model_name: str = "shibing624/text2vec-base-chinese",
        index_dir: str = 'tmp_index_dir/',
        index_name: str = "faiss.index",
        corpus_file: str = "tmp_data_dir/corpus.npy",
        num_results: int = 10,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        port: int = 8001,
):
    """main entry point of bert search backend, start the endpoints"""
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    from starlette.middleware.cors import CORSMiddleware

    print("starting boot of bert serve")
    index_file = os.path.join(index_dir, index_name)
    assert os.path.exists(index_file), f"index file {index_file} not exist"
    index = faiss.read_index(index_file)
    model = SentenceModel(model_name_or_path=model_name, device=device)
    sentences = np.load(corpus_file)
    logger.info(f'Load model success. model: {model}, index: {index}, sentences size: {len(sentences)}')

    # define the app
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    class Item(BaseModel):
        input: str = Field(..., max_length=512)

    @app.get('/')
    async def index():
        return {"message": "index, docs url: /docs"}

    @app.post('/emb')
    async def emb(item: Item):
        try:
            q = item.input
            embeddings = model.encode(q)
            result_dict = {'emb': embeddings.tolist()}
            logger.debug(f"Successfully get sentence embeddings, q:{q}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/similarity')
    async def similarity(item1: Item, item2: Item):
        try:
            q1 = item1.input
            q2 = item2.input
            emb1 = model.encode(q1)
            emb2 = model.encode(q2)
            sim_score = cos_sim(emb1, emb2)
            result_dict = {'similarity': sim_score}
            logger.debug(f"Successfully get similarity score, q1:{q1}, q2:{q2}, res: {sim_score}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/search')
    async def search(item: Item):
        try:
            q = item.input
            sorted_text_scores = _search_index(q, model, index, sentences, num_results, threshold)
            result_dict = {'result': sorted_text_scores}
            logger.debug(f"Successfully search done, q:{q}")
            return result_dict
        except Exception as e:
            logger.error(f"search error: {e}")
            return {'status': False, 'msg': e}, 400

    logger.info("Server starting!")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    """Main entry point"""
    fire.Fire(
        {
            "bert_embedding": bert_embedding,
            "bert_index": bert_index,
            "bert_filter": bert_filter,
            "bert_server": bert_server,
        }
    )


if __name__ == "__main__":
    main()
