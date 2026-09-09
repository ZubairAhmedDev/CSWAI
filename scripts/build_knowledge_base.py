from core.rag import build_index, knowledge_base_stats
if __name__=="__main__":
    build_index(batch_size=16)
    s=knowledge_base_stats()
    print("\nSUMMARY")
    print("Documents:",s["documents"])
    print("Pages:",s["pages"])
    print("Chunks:",s["chunks"])
    print("Topics:",s["topics"])
    print("EKE layers:",s["eke_layers"])
