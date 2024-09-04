from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

information = """
Linus Benedict Torvalds (/ˈliːnəs ˈtɔːrvɔːldz/ LEE-nəs TOR-vawldz,[2] Finland Swedish: [ˈliːnʉs ˈtuːrvɑlds] ⓘ; born 28 December 1969) is a Finnish-American software engineer who is the creator and lead developer of the Linux kernel. He also created the distributed version control system Git.

He was honored, along with Shinya Yamanaka, with the 2012 Millennium Technology Prize by the Technology Academy Finland "in recognition of his creation of a new open source operating system for computers leading to the widely used Linux kernel."[3] He is also the recipient of the 2014 IEEE Computer Society Computer Pioneer Award[4] and the 2018 IEEE Masaru Ibuka Consumer Electronics Award.[5]

Life and career
Early years
Torvalds was born in Helsinki, Finland, the 28th December 1969, the son of journalists Anna and Nils Torvalds,[6] the grandson of statistician Leo Törnqvist and of poet Ole Torvalds, and the great-grandson of journalist and soldier Toivo Karanko. His parents were campus radicals at the University of Helsinki in the 1960s. His family belongs to the Swedish-speaking minority in Finland. He was named after Linus Pauling, the Nobel Prize–winning American chemist, although in the book Rebel Code: Linux and the Open Source Revolution, he is quoted as saying, "I think I was named equally for Linus the Peanuts cartoon character", noting that this made him "half Nobel Prize–winning chemist and half blanket-carrying cartoon character".[7]

His interest in computers began with a VIC-20[8] at the age of 11 in 1981. He started programming for it in BASIC, then later by directly accessing the 6502 CPU in machine code (he did not utilize assembly language).[9] He then purchased a Sinclair QL, which he modified extensively, especially its operating system. "Because it was so hard to get software for it in Finland", he wrote his own assembler and editor "(in addition to Pac-Man graphics libraries)"[10] for the QL, and a few games.[11][12] He wrote a Pac-Man clone, Cool Man.

Torvalds attended the University of Helsinki from 1988 to 1996,[13] graduating with a master's degree in computer science from the NODES research group.[14] His textbooks while there included Programming the 80386[15] by John H. Crawford and Patrick P. Gelsinger, SYBEX, 1987 ISBN 0895883813, and The Design of the UNIX Operating System[16] by Maurice J. Bach, Prentice-Hall, 1986 ISBN 0-13-201799-7.[17]

He bought computer science professor Andrew Tanenbaum's book Operating Systems: Design and Implementation, in which Tanenbaum describes MINIX, an educational stripped-down version of Unix. In 1990, Torvalds resumed his university studies, and was exposed to Unix for the first time in the form of a DEC MicroVAX running ULTRIX.[18] His MSc thesis was titled Linux: A Portable Operating System.[19]

On 5 January 1991[20] he purchased an Intel 80386-based IBM PC clone[21] before receiving his MINIX copy, which in turn enabled him to begin work on Linux.

His academic career was interrupted after his first year of study when he joined the Finnish Navy Nyland Brigade in the summer of 1989, selecting the 11-month officer training program to fulfill the mandatory military service of Finland. He gained the rank of second lieutenant, with the role of an artillery observer.[22]

"""

if __name__ == "__main__":
    load_dotenv()
    print("Hello LangChain!")

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_vatiables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": information})

    print(res)
