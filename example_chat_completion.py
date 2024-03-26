# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
import time

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [{"role": "system", "content": "You are a financial analyst who is responsible for keeping a database of all deals that have been done. You will read articles and press releases about mergers and acquisitions and use your expertise to answer questions pertaining to the information."},
            {"role": "user", "content": """Novo Holdings Portfolio Company Amolyt Pharma Enters into Definitive Agreement to be Acquired by AstraZeneca NEWS PROVIDED BY Optimum Strategic Communications 14 Mar, 2024, 04:00 ET SHARE THIS ARTICLE Deal value up to US$1.05 billion Novo Holdings co-led Series A financing in Amolyt, supported additional financings and is Amolyt's largest investor Acquisition validates Novo Holdings' strategy to identify and invest long term in high quality European biotech companies COPENHAGEN, Denmark, March 14, 2024 /PRNewswire/ -- Novo Holdings A/S, a leading international life sciences investor, today announced that its portfolio company Amolyt Pharma ("Amolyt"), has entered into a definitive agreement to be acquired by AstraZeneca at a purchase price of US$800 million upfront and a potential milestone payment of US$250 million. Based in Lyon, France and Boston, US, Amolyt is building on its team's established expertise to deliver life-changing treatments to patients suffering from rare endocrine and related diseases. Amolyt's clinical pipeline includes differentiated therapeutic peptides for the treatment of underserved rare endocrine disease. Its lead asset eneboparatide (AZP-3601) is an investigational daily subcutaneous injectable parathyroid hormone receptor 1 (PTHR1) agonist for the treatment of hypoparathyroidism, that is currently in Phase 3. Novo Holdings co-led the Series A financing in 2019 when eneboparatide was still at preclinical stage, and continued to support the company through its follow-on Series B and Series C investments. This investment highlights Novo Holdings' evergreen structure and its ability to take a long-term view to advance science and medicine for the benefit of society, supporting early stage companies to advance programmes all the way through to late stage development. Naveed Siddiqi, Board Director of Amolyt and Senior Partner at Novo Holdings, said: "Amolyt is a great example of a high quality European company created to address an underserved area of high unmet medical need and led by a highly capable management team that has a clear track record of biotech success. As a long-standing shareholder and active Board member, Novo Holdings is extremely proud to support Amolyt to rapidly advance its pipeline from pre-clinical to Phase3 development including the in-licensing of additional assets in less than five years." Pierre Legault, Chairman of Amolyt Pharma's Board of Directors, added "Today's announcement recognizes the significant value we have built for our shareholders and patients. We are confident that with Alexion, Astra-Zeneca Rare Disease our innovative science will be even better positioned to ultimately reach patients with rare diseases around the world." About Novo Holdings Novo Holdings is a holding and investment company that is responsible for managing the assets and the wealth of the Novo Nordisk Foundation. The purpose of Novo Holdings is to improve people's health and the sustainability of society and the planet by generating attractive long-term returns on the assets of the Novo Nordisk Foundation. Wholly owned by the Novo Nordisk Foundation, Novo Holdings is the controlling shareholder of Novo Nordisk A/S and Novozymes A/S and manages an investment portfolio, with a long-term return perspective. Novo Holdings is a world-leading life sciences investor. Through its Seeds, Venture, Growth, and Principal Investments teams, Novo Holdings invests in life science companies at all stages of development. About Novo Holdings Venture Investments The Novo Holdings Venture Investments team is one of the largest and most active international life science venture investors with a track record of over 20 years investing in novel therapies. The Venture Investments portfolio includes both private and publicly traded investments in the biotech, medtech and digital health sectors, and spans early-stage, translatable science through commercial stage products. Novo Holdings' evergreen structure enables the Venture Investments team to take a long- term view to advance science and medicine for the benefit of society. Venture Investments operates globally with teams based in San Francisco, Boston, London, and Copenhagen. In 2023, Venture Investments returned $600 million to the Novo Nordisk Foundation, including proceeds from the sale of five portfolio companies to acquirers. About Amolyt Pharma Amolyt Pharma, a clinical stage biotechnology company, is building on its team's established expertise to deliver life-changing treatments to patients suffering from rare endocrine and related diseases. Its development portfolio includes eneboparatide (AZP-3601), a long-acting PTH1 receptor agonist as a potential treatment for hypoparathyroidism, and AZP-3813, a peptide growth hormone receptor antagonist for the potential treatment of acromegaly. Amolyt Pharma aims to further expand and develop its portfolio by leveraging its global network in the field of endocrinology and with support from a strong syndicate of international investors. To learn more, visit https://amolytpharma.com/ or follow us on Twitter and LinkedIn. SOURCE Optimum Strategic Communications --END PRESS RELEASE-- Your task is to record all useful data from this press release to be recorded in you database. Key information to record includes the following: -Name of Company Sold a.k.a ASSET -Name of Company Selling the asset a.k.a. SELLER -Name of Buyer of the asset a.k.a BUYER -Size of the transaction -Rationale of why the BUYER bought the ASSET -Names of any important people in the companies -Descriptions of the Companies involved -Any extra notes that you might think could be valuable to the dataset Record your data down as a json object."""}]]
        

    start = time.time()
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        load_start_state = True,
        #save_start_state = True
    )
    end = time.time()
    elapsed = end - start

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")
    print(f"runtime: {elapsed:.5f} seconds")


if __name__ == "__main__":
    fire.Fire(main)
