from hashlib import sha256
from typing import Any, Collection, Literal, Optional, Set, Union

from lightrag.cleaner.clean_processor import CleanProcessor
from lightrag.splitter.document import Document
from lightrag.splitter.text_splitter import TS, RecursiveCharacterTextSplitter, TokenTextSplitter
from lightrag.tokenizers.gpt2_tokenzier import GPT2Tokenizer


class EnhanceRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """
    This class is used to implement from_gpt2_encoder, to prevent using of tiktoken
    """

    @classmethod
    def from_encoder(
        cls: type[TS],
        allowed_special: Union[Literal[all], Set[str]] = set(),
        disallowed_special: Union[Literal[all], Collection[str]] = "all",
        **kwargs: Any,
    ):
        def _token_encoder(text: str) -> int:
            if not text:
                return 0

            return GPT2Tokenizer.get_num_tokens(text)

        if issubclass(cls, TokenTextSplitter):
            extra_kwargs = {
                "model_name": "gpt2",
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_token_encoder, **kwargs)


class FixedRecursiveCharacterTextSplitter(EnhanceRecursiveCharacterTextSplitter):
    def __init__(
        self,
        fixed_separator: str = "\n\n",
        separators: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._fixed_separator = fixed_separator
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """Split incoming text and return chunks."""
        if self._fixed_separator:
            chunks = text.split(self._fixed_separator)
        else:
            chunks = [text]

        final_chunks = []
        for chunk in chunks:
            if self._length_function(chunk) > self._chunk_size:
                final_chunks.extend(self.recursive_split_text(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def recursive_split_text(self, text: str) -> list[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = self._separators[-1]
        for _s in self._separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        # Now that we have the separator, split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _good_splits_lengths = []  # cache the lengths of the splits
        for s in splits:
            s_len = self._length_function(s)
            if s_len < self._chunk_size:
                _good_splits.append(s)
                _good_splits_lengths.append(s_len)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, separator, _good_splits_lengths)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                    _good_splits_lengths = []
                other_info = self.recursive_split_text(s)
                final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, separator, _good_splits_lengths)
            final_chunks.extend(merged_text)
        return final_chunks


def generate_text_hash(text: str) -> str:
    hash_text = str(text) + "None"
    return sha256(hash_text.encode()).hexdigest()


if __name__ == "__main__":
    splitter = EnhanceRecursiveCharacterTextSplitter.from_encoder(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "。", ". ", " ", ""]
    )

    doc = Document(
        page_content="""中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
    )
    document_text = CleanProcessor.clean(
        doc.page_content,
        {"rules": {"pre_processing_rules": [{"id": "remove_extra_spaces", "enabled": True}]}},
    )
    doc.page_content = document_text

    document_nodes = splitter.split_documents([doc])
    split_documents = []
    for index, document_node in enumerate(document_nodes):
        item = {}

        if document_node.page_content.strip():
            page_content = document_node.page_content

            item["tokens"] = GPT2Tokenizer.get_num_tokens(page_content)
            item["chunk_order_index"] = index
            # delete Splitter character
            if page_content.startswith(".") or page_content.startswith("。"):
                page_content = page_content[1:].strip()
            else:
                page_content = page_content

            if len(page_content) > 0:
                item["content"] = page_content
                split_documents.append(item)

    for document in split_documents:
        print(document["tokens"])
        print(document["content"])
