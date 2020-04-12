/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
* Extended to include the implementation of Transformer-HAN-encoder architecture by Sameen Maruf
*/

#pragma once

// All utilities
#include "utils.h"

// Layers
#include "layers.h"

// Base Transformer
#include "transformer.h"

using namespace std;
using namespace dynet;

namespace transformer {

//--- Encoder Context Layer
struct EncoderContext{
    explicit EncoderContext(DyNetModel* mod, TransformerConfig& tfc, Encoder* p_encoder)
            : _word_attention_sublayer(mod, tfc)
            , _sent_attention_sublayer(mod, tfc)
            , _feed_forward_sublayer(mod, tfc)
    {
        // for layer normalisation
        _p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
        _p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
        _p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
        _p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));

        // for context gating
        _p_Cs = mod->add_parameters({tfc._num_units, tfc._num_units});
        _p_Csc = mod->add_parameters({tfc._num_units, tfc._num_units});

        _p_tfc = &tfc;

        _p_encoder = p_encoder;
    }

    ~EncoderContext(){}

    //for masking the words due to padding
    MaskSent _word_mask_minus1, _word_mask_minus2, _word_mask_plus1, _word_mask_plus2, _word_mask;

    // multi-head attention sub-layer for the word-level context
    MultiHeadWordAttentionLayer _word_attention_sublayer;

    // multi-head attention sub-layer for the sent-level context
    MultiHeadSentAttentionLayer _sent_attention_sublayer;

    // position-wise feed forward sub-layer
    FeedForwardLayer _feed_forward_sublayer;

    // for layer normalisation
    dynet::Parameter _p_ln1_g, _p_ln1_b;// layer normalisation 1
    dynet::Parameter _p_ln2_g, _p_ln2_b;// layer normalisation 2

    // for context gating
    dynet::Parameter _p_Cs, _p_Csc;

    // transformer config pointer
    TransformerConfig* _p_tfc = nullptr;

    // encoder object pointer
    Encoder* _p_encoder = nullptr;

    dynet::Expression compute_wordrep_and_masks_minus1(dynet::ComputationGraph &cg
            , vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids, vector<unsigned int>& sent_size)
    {
        unsigned bsize = sids.size();

        vector<int> sids_minus1;//save index for previous sentence
        unsigned max_clen = 3;//get maximum length for previous sentence in batch including the bos and eos tokens
        for (unsigned bs = 0; bs < bsize; ++bs) {
            sids_minus1.push_back(sids[bs] - 1);

            if (sids_minus1[bs] >= 0)
                max_clen = std::max(max_clen, sent_size[sids_minus1[bs]]);
        }

        //dynet::Expression input_zeros = dynet::zeros(cg, {_p_tfc->_num_units});

        std::vector<std::vector<float>> v_seq_masks(bsize);
        std::vector<Expression> word_rep(bsize);
        vector<Expression> batch_word_embeddings;

        for (unsigned l = 1; l < max_clen - 1; l++){
            for (unsigned bs = 0; bs < bsize; ++bs){
                if (sids_minus1[bs] >= 0) {
                    if (l < sent_size[sids_minus1[bs]] - 1) {//subtract one to avoid eos token representation for sentence whose length is not maximum
                        word_rep[bs] = dynet::input(cg, {_p_tfc->_num_units}, srcwordrep_doc[sids_minus1[bs]][l]);
                        v_seq_masks[bs].push_back(0.f);// padding position
                    } else {
                        word_rep[bs] = dynet::lookup(cg, _p_encoder->_p_embed_s, (unsigned)_p_tfc->_sm._kSRC_EOS);
                        v_seq_masks[bs].push_back(1.f);// padding position
                    }
                }
                else {
                    word_rep[bs] = dynet::lookup(cg, _p_encoder->_p_embed_s, (unsigned)_p_tfc->_sm._kSRC_EOS);
                    v_seq_masks[bs].push_back(1.f);// padding position
                }
            }

            batch_word_embeddings.push_back(dynet::concatenate_to_batch(word_rep));
        }
        dynet::Expression i_word_ctx = dynet::concatenate_cols(batch_word_embeddings);// ((num_units, Lc), batch_size)

        _word_mask_minus1.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
        _word_mask_minus1.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else
        _word_mask_minus1.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif

        return i_word_ctx;
    }

    dynet::Expression compute_wordrep_and_masks_plus1(dynet::ComputationGraph &cg
            , vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids, vector<unsigned int>& sent_size)
    {
        unsigned bsize = sids.size();

        vector<unsigned int> sids_plus1;//save index for next sentence
        unsigned max_clen = 3;//get maximum length for next sentence in batch including the bos and eos tokens
        for (unsigned bs = 0; bs < bsize; ++bs) {
            sids_plus1.push_back(sids[bs] + 1);

            if (sids_plus1[bs] < srcwordrep_doc.size())
                max_clen = std::max(max_clen, sent_size[sids_plus1[bs]]);
        }
        //dynet::Expression input_zeros = dynet::zeros(cg, {_p_tfc->_num_units});

        std::vector<std::vector<float>> v_seq_masks(bsize);
        std::vector<Expression> word_rep(bsize);
        vector<Expression> batch_word_embeddings;

        for (unsigned l = 1; l < max_clen - 1; l++){
            for (unsigned bs = 0; bs < bsize; ++bs){
                if (sids_plus1[bs] < srcwordrep_doc.size()) {
                    if (l < sent_size[sids_plus1[bs]] - 1) {//subtract one to avoid eos token representation for sentence whose length is not maximum
                        word_rep[bs] = dynet::input(cg, {_p_tfc->_num_units}, srcwordrep_doc[sids_plus1[bs]][l]);
                        v_seq_masks[bs].push_back(0.f);// padding position
                    } else {
                        word_rep[bs] = dynet::lookup(cg, _p_encoder->_p_embed_s, (unsigned)_p_tfc->_sm._kSRC_EOS);
                        v_seq_masks[bs].push_back(1.f);// padding position
                    }
                }
                else {
                    word_rep[bs] = dynet::lookup(cg, _p_encoder->_p_embed_s, (unsigned)_p_tfc->_sm._kSRC_EOS);
                    v_seq_masks[bs].push_back(1.f);// padding position
                }
            }

            batch_word_embeddings.push_back(dynet::concatenate_to_batch(word_rep));
        }
        dynet::Expression i_word_ctx = dynet::concatenate_cols(batch_word_embeddings);// ((num_units, Lc), batch_size)

        _word_mask_plus1.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
        _word_mask_plus1.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else
        _word_mask_plus1.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif

        return i_word_ctx;
    }

    dynet::Expression compute_wordrep_and_masks(dynet::ComputationGraph &cg
            , vector<vector<dynet::real>> srcwordrep_sent)
    {
        unsigned bsize = 1;

        vector<int> sids_minus1;//save index for previous sentence
        unsigned max_clen = srcwordrep_sent.size();//get maximum length for previous sentence in batch including the bos and eos tokens

        std::vector<std::vector<float>> v_seq_masks(bsize);
        std::vector<Expression> word_rep(bsize);
        vector<Expression> batch_word_embeddings;

        for (unsigned l = 1; l < max_clen - 1; l++){
            for (unsigned bs = 0; bs < bsize; ++bs){
                word_rep[bs] = dynet::input(cg, {_p_tfc->_num_units}, srcwordrep_sent[l]);
                v_seq_masks[bs].push_back(0.f);// padding position
            }

            batch_word_embeddings.push_back(dynet::concatenate_to_batch(word_rep));
        }
        dynet::Expression i_word_ctx = dynet::concatenate_cols(batch_word_embeddings);// ((num_units, Lc), batch_size)

        _word_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
        _word_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else
        _word_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif

        return i_word_ctx;
    }

    dynet::Expression compute_wordrep_and_masks_minus2(dynet::ComputationGraph &cg
            , vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids, vector<unsigned int>& sent_size)
    {
        unsigned bsize = sids.size();

        vector<int> sids_minus2;//save index for previous 2nd sentence
        unsigned max_clen = 3;//get maximum length for previous 2nd sentence in batch including the bos and eos tokens
        for (unsigned bs = 0; bs < bsize; ++bs) {
            sids_minus2.push_back(sids[bs] - 2);

            if (sids_minus2[bs] >= 0)
                max_clen = std::max(max_clen, sent_size[sids_minus2[bs]]);
        }

        //dynet::Expression input_zeros = dynet::zeros(cg, {_p_tfc->_num_units});

        std::vector<std::vector<float>> v_seq_masks(bsize);
        std::vector<Expression> word_rep(bsize);
        vector<Expression> batch_word_embeddings;

        for (unsigned l = 1; l < max_clen - 1; l++){
            for (unsigned bs = 0; bs < bsize; ++bs){
                if (sids_minus2[bs] >= 0) {
                    if (l < sent_size[sids_minus2[bs]] - 1) {
                        word_rep[bs] = dynet::input(cg, {_p_tfc->_num_units}, srcwordrep_doc[sids_minus2[bs]][l]);
                        v_seq_masks[bs].push_back(0.f);// padding position
                    } else {
                        word_rep[bs] = dynet::lookup(cg, _p_encoder->_p_embed_s, (unsigned)_p_tfc->_sm._kSRC_EOS);
                        v_seq_masks[bs].push_back(1.f);// padding position
                    }
                }
                else {
                    word_rep[bs] = dynet::lookup(cg, _p_encoder->_p_embed_s, (unsigned)_p_tfc->_sm._kSRC_EOS);
                    v_seq_masks[bs].push_back(1.f);// padding position
                }
            }

            batch_word_embeddings.push_back(dynet::concatenate_to_batch(word_rep));
        }
        dynet::Expression i_word_ctx = dynet::concatenate_cols(batch_word_embeddings);// ((num_units, Lc), batch_size)

        _word_mask_minus2.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
        _word_mask_minus2.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else
        _word_mask_minus2.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif

        return i_word_ctx;
    }

    dynet::Expression compute_wordrep_and_masks_plus2(dynet::ComputationGraph &cg
            , vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids, vector<unsigned int>& sent_size)
    {
        unsigned bsize = sids.size();

        vector<unsigned int> sids_plus2;//save index for next sentence
        unsigned max_clen = 3;//get maximum length for next sentence in batch including the bos and eos tokens
        for (unsigned bs = 0; bs < bsize; ++bs) {
            sids_plus2.push_back(sids[bs] + 2);

            if (sids_plus2[bs] < srcwordrep_doc.size())
                max_clen = std::max(max_clen, sent_size[sids_plus2[bs]]);
        }
        //dynet::Expression input_zeros = dynet::zeros(cg, {_p_tfc->_num_units});

        std::vector<std::vector<float>> v_seq_masks(bsize);
        std::vector<Expression> word_rep(bsize);
        vector<Expression> batch_word_embeddings;

        for (unsigned l = 1; l < max_clen - 1; l++){
            for (unsigned bs = 0; bs < bsize; ++bs){
                if (sids_plus2[bs] < srcwordrep_doc.size()) {
                    if (l < sent_size[sids_plus2[bs]] - 1) {//subtract one to avoid eos token representation for sentence whose length is not maximum
                        word_rep[bs] = dynet::input(cg, {_p_tfc->_num_units}, srcwordrep_doc[sids_plus2[bs]][l]);
                        v_seq_masks[bs].push_back(0.f);// padding position
                    } else {
                        word_rep[bs] = dynet::lookup(cg, _p_encoder->_p_embed_s, (unsigned)_p_tfc->_sm._kSRC_EOS);
                        v_seq_masks[bs].push_back(1.f);// padding position
                    }
                }
                else {
                    word_rep[bs] = dynet::lookup(cg, _p_encoder->_p_embed_s, (unsigned)_p_tfc->_sm._kSRC_EOS);
                    v_seq_masks[bs].push_back(1.f);// padding position
                }
            }

            batch_word_embeddings.push_back(dynet::concatenate_to_batch(word_rep));
        }
        dynet::Expression i_word_ctx = dynet::concatenate_cols(batch_word_embeddings);// ((num_units, Lc), batch_size)

        _word_mask_plus2.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
        _word_mask_plus2.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else
        _word_mask_plus2.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif

        return i_word_ctx;
    }

    dynet::Expression build_wordgraph(dynet::ComputationGraph &cg
            , const dynet::Expression& i_src, const dynet::Expression& i_word_ctx, const MaskSent& word_mask)
    {
        // multi-head attention sub-layer
        dynet::Expression i_word_att = _word_attention_sublayer.build_graph(cg, i_src, i_word_ctx, word_mask);// ((num_units, Lx), batch_size)

        // dropout to the above sub-layer
        if (_p_tfc->_use_dropout && _p_tfc->_encoder_sublayer_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
            i_word_att = dynet::dropout_dim(i_word_att, 1/*col-major*/,
                                             _p_tfc->_encoder_sublayer_dropout_rate);// col-wise dropout
#else
        i_word_att = dynet::dropout(i_word_att, _p_tfc->_encoder_sublayer_dropout_rate);// full dropout
#endif

        return i_word_att;
    }

    dynet::Expression build_graph(dynet::ComputationGraph &cg
            , const dynet::Expression& i_src, vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids)
    {
        // get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b
        dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
        dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
        dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
        dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);

        //get expressions for context gating
        dynet::Expression i_Cs = dynet::parameter(cg, _p_Cs);
        dynet::Expression i_Csc = dynet::parameter(cg, _p_Csc);

        //to save the number of tokens in each source sentence
        vector<unsigned int> dslen;
        for (unsigned i = 0; i < srcwordrep_doc.size(); i++) {
            vector<vector<dynet::real>> wordrep_sent = srcwordrep_doc[i];
            dslen.push_back(wordrep_sent.size());//contains bos and eos tokens
        }

        //for previous/next sentence
        dynet::Expression i_word_ctx_minus1, i_word_att_minus1, i_word_ctx_plus1, i_word_att_plus1;
        dynet::Expression i_word_ctx_minus2, i_word_att_minus2, i_word_ctx_plus2, i_word_att_plus2;
        dynet::Expression i_ctx_plus1, i_ctx_minus1, i_ctx_plus2, i_ctx_minus2;
        dynet::Expression i_sent_att;

        if (_p_tfc->_context_type == 1) {
            i_word_ctx_plus1 = compute_wordrep_and_masks_plus1(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_plus1 = build_wordgraph(cg, i_src, i_word_ctx_plus1, _word_mask_plus1);

            // position-wise layer normalisation 1
            i_sent_att = layer_norm_colwise_3(i_word_att_plus1, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)
        }
        else if (_p_tfc->_context_type == 2) {
            i_word_ctx_minus1 = compute_wordrep_and_masks_minus1(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_minus1 = build_wordgraph(cg, i_src, i_word_ctx_minus1, _word_mask_minus1);

            // position-wise layer normalisation 1
            i_sent_att = layer_norm_colwise_3(i_word_att_minus1, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)
        }
        else if (_p_tfc->_context_type == 3) {
            i_word_ctx_minus1 = compute_wordrep_and_masks_minus1(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_minus1 = build_wordgraph(cg, i_src, i_word_ctx_minus1, _word_mask_minus1);

            // position-wise layer normalisation 1
            i_ctx_minus1 = layer_norm_colwise_3(i_word_att_minus1, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            i_word_ctx_plus1 = compute_wordrep_and_masks_plus1(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_plus1 = build_wordgraph(cg, i_src, i_word_ctx_plus1, _word_mask_plus1);

            // position-wise layer normalisation 1
            i_ctx_plus1 = layer_norm_colwise_3(i_word_att_plus1, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            //Sentence-level attention
            auto& d = i_ctx_plus1.dim();//(num_units, Lx), batch_size

            dynet::Expression i_x1, i_x2, i_x_ctx;
            std::vector<dynet::Expression> vsent_att(d[1]);
            for (unsigned i = 0; i < d[1]; ++i) {
                i_x1 = dynet::pick(i_ctx_minus1, i, 1);
                i_x2 = dynet::pick(i_ctx_plus1, i, 1);

                i_x_ctx = dynet::concatenate_cols({i_x1, i_x2});

                // multi-head attention sub-layer
                vsent_att[i] = _sent_attention_sublayer.build_graph(cg, dynet::reshape(dynet::pick(i_src, i, 1), {d[0], 1}), i_x_ctx);// ((num_units, 1), batch_size)
            }
            i_sent_att = dynet::concatenate_cols(vsent_att);
        }
        else if (_p_tfc->_context_type == 4) {
            i_word_ctx_plus1 = compute_wordrep_and_masks_plus1(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_plus1 = build_wordgraph(cg, i_src, i_word_ctx_plus1, _word_mask_plus1);

            // position-wise layer normalisation 1
            i_ctx_plus1 = layer_norm_colwise_3(i_word_att_plus1, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            i_word_ctx_plus2 = compute_wordrep_and_masks_plus2(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_plus2 = build_wordgraph(cg, i_src, i_word_ctx_plus2, _word_mask_plus2);

            // position-wise layer normalisation 1
            i_ctx_plus2 = layer_norm_colwise_3(i_word_att_plus2, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            //Sentence-level attention
            auto& d = i_ctx_plus1.dim();//(num_units, Lx), batch_size

            dynet::Expression i_x1, i_x2, i_x_ctx;
            std::vector<dynet::Expression> vsent_att(d[1]);
            for (unsigned i = 0; i < d[1]; ++i) {
                i_x1 = dynet::pick(i_ctx_plus1, i, 1);
                i_x2 = dynet::pick(i_ctx_plus2, i, 1);

                i_x_ctx = dynet::concatenate_cols({i_x1, i_x2});

                // multi-head attention sub-layer
                vsent_att[i] = _sent_attention_sublayer.build_graph(cg, dynet::reshape(dynet::pick(i_src, i, 1), {d[0], 1}), i_x_ctx);// ((num_units, 1), batch_size)
            }
            i_sent_att = dynet::concatenate_cols(vsent_att);
        }
        else if (_p_tfc->_context_type == 5) {
            i_word_ctx_minus2 = compute_wordrep_and_masks_minus2(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_minus2 = build_wordgraph(cg, i_src, i_word_ctx_minus2, _word_mask_minus2);

            // position-wise layer normalisation 1
            i_ctx_minus2 = layer_norm_colwise_3(i_word_att_minus2, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            i_word_ctx_minus1 = compute_wordrep_and_masks_minus1(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_minus1 = build_wordgraph(cg, i_src, i_word_ctx_minus1, _word_mask_minus1);

            // position-wise layer normalisation 1
            i_ctx_minus1 = layer_norm_colwise_3(i_word_att_minus1, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            //Sentence-level attention
            auto& d = i_ctx_minus1.dim();//(num_units, Lx), batch_size

            dynet::Expression i_x1, i_x2, i_x_ctx;
            std::vector<dynet::Expression> vsent_att(d[1]);
            for (unsigned i = 0; i < d[1]; ++i) {
                i_x1 = dynet::pick(i_ctx_minus2, i, 1);
                i_x2 = dynet::pick(i_ctx_minus1, i, 1);

                i_x_ctx = dynet::concatenate_cols({i_x1, i_x2});

                // multi-head attention sub-layer
                vsent_att[i] = _sent_attention_sublayer.build_graph(cg, dynet::reshape(dynet::pick(i_src, i, 1), {d[0], 1}), i_x_ctx);// ((num_units, 1), batch_size)
            }
            i_sent_att = dynet::concatenate_cols(vsent_att);
        }
        else if (_p_tfc->_context_type == 6) {
            i_word_ctx_minus2 = compute_wordrep_and_masks_minus2(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_minus2 = build_wordgraph(cg, i_src, i_word_ctx_minus2, _word_mask_minus2);

            // position-wise layer normalisation 1
            i_ctx_minus2 = layer_norm_colwise_3(i_word_att_minus2, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            i_word_ctx_minus1 = compute_wordrep_and_masks_minus1(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_minus1 = build_wordgraph(cg, i_src, i_word_ctx_minus1, _word_mask_minus1);

            // position-wise layer normalisation 1
            i_ctx_minus1 = layer_norm_colwise_3(i_word_att_minus1, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            i_word_ctx_plus1 = compute_wordrep_and_masks_plus1(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_plus1 = build_wordgraph(cg, i_src, i_word_ctx_plus1, _word_mask_plus1);

            // position-wise layer normalisation 1
            i_ctx_plus1 = layer_norm_colwise_3(i_word_att_plus1, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            i_word_ctx_plus2 = compute_wordrep_and_masks_plus2(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
            i_word_att_plus2 = build_wordgraph(cg, i_src, i_word_ctx_plus2, _word_mask_plus2);

            // position-wise layer normalisation 1
            i_ctx_plus2 = layer_norm_colwise_3(i_word_att_plus2, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

            //Sentence-level attention
            auto& d = i_ctx_minus2.dim();//(num_units, Lx), batch_size

            dynet::Expression i_x1, i_x2, i_x3, i_x4, i_x_ctx;
            std::vector<dynet::Expression> vsent_att(d[1]);
            for (unsigned i = 0; i < d[1]; ++i) {
                i_x1 = dynet::pick(i_ctx_minus2, i, 1);
                i_x2 = dynet::pick(i_ctx_minus1, i, 1);
                i_x3 = dynet::pick(i_ctx_plus1, i, 1);
                i_x4 = dynet::pick(i_ctx_plus2, i, 1);

                i_x_ctx = dynet::concatenate_cols({i_x1, i_x2, i_x3, i_x4});

                // multi-head attention sub-layer
                vsent_att[i] = _sent_attention_sublayer.build_graph(cg, dynet::reshape(dynet::pick(i_src, i, 1), {d[0], 1}), i_x_ctx);// ((num_units, 1), batch_size)
            }
            i_sent_att = dynet::concatenate_cols(vsent_att);
        }

        // position-wise feed-forward sub-layer
        dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_sent_att);// ((num_units, Lx), batch_size)

        dynet::Expression i_ctxl = i_ff;

        //context gating
        dynet:: Expression gate_input = i_Cs * i_src + i_Csc * i_ctxl;
        dynet::Expression lambda = logistic(gate_input);
        dynet::Expression i_htilde_t = cmult(lambda, i_src) + cmult(1.f - lambda, i_ctxl);

        // position-wise layer normalisation 2
        i_htilde_t = layer_norm_colwise_3(i_htilde_t, i_ln2_g, i_ln2_b);// ((num_units, Lx), batch_size)

        return i_htilde_t;
    }

    dynet::Expression build_graph(dynet::ComputationGraph &cg
            , const dynet::Expression& i_src, vector<vector<dynet::real>> srcwordrep_sent)
    {
        // get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b
        dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
        dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
        dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
        dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);

        //get expressions for context gating
        dynet::Expression i_Cs = dynet::parameter(cg, _p_Cs);
        dynet::Expression i_Csc = dynet::parameter(cg, _p_Csc);

        //for previous/next sentence
        dynet::Expression i_word_ctx1, i_word_att1;
        i_word_ctx1 = compute_wordrep_and_masks(cg, srcwordrep_sent);//((num_units, dw), batch_size) here dw is the max number of tokens in the context sentences in batch
	//auto& dc = i_word_ctx1.dim();
	//auto bs = dc.batch_elems();
        //cout << "Dimensions of context matrix: " << dc[0] << " x " << dc[1] << " x " << bs << endl;

        i_word_att1 = build_wordgraph(cg, i_src, i_word_ctx1, _word_mask);

        // position-wise layer normalisation 1
        dynet::Expression i_ctx1 = layer_norm_colwise_3(i_word_att1, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

        // position-wise feed-forward sub-layer
        dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_ctx1);// ((num_units, Lx), batch_size)

        dynet::Expression i_ctxl = i_ff;

        //context gating
        dynet:: Expression gate_input = i_Cs * i_src + i_Csc * i_ctxl;
        dynet::Expression lambda = logistic(gate_input);
        dynet::Expression i_htilde_t = cmult(lambda, i_src) + cmult(1.f - lambda, i_ctxl);

        // position-wise layer normalisation 2
        i_htilde_t = layer_norm_colwise_3(i_htilde_t, i_ln2_g, i_ln2_b);// ((num_units, Lx), batch_size)

        return i_htilde_t;
    }
};
typedef std::shared_ptr<EncoderContext> EncoderContextPointer;
//---

//--- Transformer Model w/Context
struct TransformerContextModel {

public:
	explicit TransformerContextModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td);

	explicit TransformerContextModel();

	~TransformerContextModel(){}

	// for initialisation
	void initialise(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td);

	// for training	
    dynet::Expression build_graph(dynet::ComputationGraph &cg
            , const WordIdSentences& ssents/*batched*/
            , const WordIdSentences& tsents/*batched*/
            , vector<vector<vector<dynet::real>>> wordrep_doc
            , vector<unsigned int> sids
            , ModelStats* pstats=nullptr
            , bool is_eval_on_dev=false);
    //for getting representations at decoding time
    std::vector<std::vector<dynet::real>> compute_source_rep(dynet::ComputationGraph &cg, const WordIdSentence& sent);
    dynet::Expression compute_source_rep(dynet::ComputationGraph &cg
            , const WordIdSentence& sent, std::vector<std::vector<dynet::real>>& srcword_rep);// source representation given real sources
    // for decoding
	dynet::Expression compute_source_rep(dynet::ComputationGraph &cg
		, const WordIdSentences& ssents);// source representation given real sources
	dynet::Expression step_forward(dynet::ComputationGraph & cg
		, const dynet::Expression& i_src_rep
		, const WordIdSentence &partial_sent
		, bool log_prob
		, std::vector<dynet::Expression> &aligns
		, float sm_temp=1.f);// forward step to get softmax scores    
    dynet::Expression step_forward(dynet::ComputationGraph &cg
		, const dynet::Expression& i_src_rep
		, const WordIdSentences &partial_sents
		, bool log_prob
		, std::vector<dynet::Expression> &aligns
		, float sm_temp=1.f);
	dynet::Expression step_forward(dynet::ComputationGraph &cg
		, const dynet::Expression& i_src_rep
		, std::vector<Expression>& v_soft_targets
		, float sm_temp=1.f);	
    WordIdSentence greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, unsigned length_ratio);
    WordIdSentence greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, vector<vector<dynet::real>> wordrep_sent, unsigned length_ratio);
    WordIdSentence greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, vector<vector<vector<dynet::real>>> wordrep_doc,
                                 vector<unsigned int> sids, unsigned length_ratio);
    
    dynet::ParameterCollection& get_model_parameters();
    dynet::ParameterCollection& get_context_model_parameters();
    void initialise_baseparams_from_file(const std::string &params_file);
    void initialise_params_from_file(const std::string &params_file);
    void save_params_to_file(const std::string &params_file);

	void set_dropout(bool is_activated = true);

	dynet::Dict& get_source_dict();
	dynet::Dict& get_target_dict();

	TransformerConfig& get_config();

protected:

    DyNetModel *_model;
    DyNetModel _base_model;
    DyNetModel _context_model;

	DyNetModelPointer _all_params;// all model parameters live in this object pointer. This object will be automatically released once unused!
    DyNetModelPointer _base_params;
    DyNetModelPointer _context_params;

	EncoderPointer _encoder;// encoder
	DecoderPointer _decoder;// decoder

    EncoderContextPointer _encoder_context;
    //DecoderContextPointer _decoder_context;

	std::pair<dynet::Dict, dynet::Dict> _dicts;// pair of source and target vocabularies

	dynet::Parameter _p_Wo_bias;// bias of final linear projection layer

	TransformerConfig _tfc;// local configuration storage
};

TransformerContextModel::TransformerContextModel(){
    _all_params = nullptr;
    _base_params = nullptr;
    _context_params = nullptr;

	_encoder = nullptr;
	_decoder = nullptr;

    _encoder_context = nullptr;
    //_decoder_context = nullptr;
}

TransformerContextModel::TransformerContextModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td)
: _tfc(tfc)
{
    _model = new DyNetModel();
    _all_params.reset(_model);// create new model parameter object

    _base_model = _model->add_subcollection("transformer");
    _base_params.reset(&_base_model);
    _context_model = _model->add_subcollection("context");
    _context_params.reset(&_context_model);

	_encoder.reset(new Encoder(_base_params.get(), _tfc));// create new encoder object

	_decoder.reset(new Decoder(_base_params.get(), _tfc, _encoder.get()));// create new decoder object

    _encoder_context.reset(new EncoderContext(_context_params.get(), _tfc, _encoder.get()));//create new encoder context object

    //_decoder_context.reset(new DecoderContext(_context_params.get(), _tfc, _decoder.get()));//create new decoder context object

	// final output projection layer
	_p_Wo_bias = _base_params.get()->add_parameters({_tfc._tgt_vocab_size});// optional

	// dictionaries
	_dicts.first = sd;
	_dicts.second = td;
}

void TransformerContextModel::initialise(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td)
{
    _tfc = tfc;

    _model = new DyNetModel();
    _all_params.reset(_model);// create new model parameter object

    _base_model = _model->add_subcollection("transformer");
    _base_params.reset(&_base_model);
    _context_model = _model->add_subcollection("context");
    _context_params.reset(&_context_model);

    _encoder.reset(new Encoder(_base_params.get(), _tfc));// create new encoder object

    _decoder.reset(new Decoder(_base_params.get(), _tfc, _encoder.get()));// create new decoder object

    _encoder_context.reset(new EncoderContext(_context_params.get(), _tfc, _encoder.get()));//create new encoder context object

    //_decoder_context.reset(new DecoderContext(_context_params.get(), _tfc, _decoder.get()));//create new decoder context object

    // final output projection layer
    _p_Wo_bias = _base_params.get()->add_parameters({_tfc._tgt_vocab_size});// optional

    // dictionaries
    _dicts.first = sd;
    _dicts.second = td;
}

dynet::Expression TransformerContextModel::compute_source_rep(dynet::ComputationGraph &cg
	, const WordIdSentences& ssents)// for decoding only
{
	// encode source
	return _encoder.get()->build_graph(cg, ssents);// ((num_units, Lx), batch_size)
}

//for getting representations
dynet::Expression TransformerContextModel::compute_source_rep(dynet::ComputationGraph &cg
        , const WordIdSentence& sent, std::vector<std::vector<dynet::real>>& srcword_rep)
{
    //encode source
    dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, WordIdSentences(1, sent));//(num_units, Lx)
    std::vector<dynet::real> i_src_td = as_vector(cg.get_value(i_src_ctx));//(num_units x Lx) x 1 i.e. flattens the tensor

    auto& d = i_src_ctx.dim();
    unsigned b = d[0] * d[1];
    unsigned steps = d[0];

    //cout << d[0] << " and " << d[1] << endl;
    unsigned t;
    for (unsigned i = 0; i < b; i+=steps){//to recreate the matrix containing representations
        t = i / steps;
        std::vector<dynet::real> word_rep(i_src_td.begin() + i, i_src_td.begin() + i + steps);
        srcword_rep[t] = word_rep;
    }

    return i_src_ctx;
}

//for source-side representations from the encoder
std::vector<std::vector<dynet::real>> TransformerContextModel::compute_source_rep(dynet::ComputationGraph &cg
        , const WordIdSentence& sent)
{
    std::vector<std::vector<dynet::real>> srcword_rep(sent.size());

    //encode source
    dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, WordIdSentences(1, sent));//(num_units, Lx)
    std::vector<dynet::real> i_src_td = as_vector(cg.forward(i_src_ctx));//(num_units x Lx) x 1 i.e. flattens the tensor

    auto& d = i_src_ctx.dim();
    unsigned b = d[0] * d[1];
    unsigned steps = d[0];

    //cout << d[0] << " and " << d[1] << endl;
    unsigned t;
    for (unsigned i = 0; i < b; i+=steps){//to recreate the matrix containing representations
        t = i / steps;
        std::vector<dynet::real> word_rep(i_src_td.begin() + i, i_src_td.begin() + i + steps);
        srcword_rep[t] = word_rep;
    }

    return srcword_rep;
}

dynet::Expression TransformerContextModel::step_forward(dynet::ComputationGraph &cg
	, const dynet::Expression& i_src_rep
	, const WordIdSentence &partial_sent
	, bool log_prob
	, std::vector<dynet::Expression> &aligns
	, float sm_temp)
{
	// decode target
	// IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, WordIdSentences(1, partial_sent), i_src_rep);// the whole matrix of context representation for every words in partial_sent - which is also wasteful because we only need the representation of last comlumn?

	// only consider the prediction of last column in the matrix
	dynet::Expression i_tgt_t;
	if (partial_sent.size() == 1) i_tgt_t = i_tgt_ctx;
	else 
		//i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
		i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(partial_sent.size() - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

	// output linear projections (w/ bias)
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
	dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

	// FIXME: get the alignments for visualisation
	// ToDo

	// compute softmax prediction
	if (log_prob)
		return dynet::log_softmax(i_r_t / sm_temp);// log_softmax w/ temperature
	else
		return dynet::softmax(i_r_t / sm_temp);// softmax w/ temperature
}

// batched version
dynet::Expression TransformerContextModel::step_forward(dynet::ComputationGraph &cg
	, const dynet::Expression& i_src_rep
	, const WordIdSentences &partial_sents
	, bool log_prob
	, std::vector<dynet::Expression> &aligns
	, float sm_temp)
{
	// decode target
	// IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, partial_sents, i_src_rep);// the whole matrix of context representation for every words in partial_sents is computed - which is also wasteful because we only need the representation of last comlumn?

	// only consider the prediction of last column in the matrix	
	dynet::Expression i_tgt_t;
	if (_decoder.get()->_batch_tlen == 1) i_tgt_t = i_tgt_ctx;
	else 
		//i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
		i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(_decoder.get()->_batch_tlen - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

	// output linear projections (w/ bias)
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
	dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

	// FIXME: get the alignments for visualisation
	// ToDo

	// compute softmax prediction (note: return a batch of softmaxes)
	if (log_prob)
		return dynet::log_softmax(i_r_t / sm_temp);// log_softmax w/ temperature
	else
		return dynet::softmax(i_r_t / sm_temp);// softmax w/ temperature
}

dynet::Expression TransformerContextModel::step_forward(dynet::ComputationGraph &cg
		, const dynet::Expression& i_src_rep
		, std::vector<Expression>& v_soft_targets
		, float sm_temp)
{
	// decode target
	// IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, v_soft_targets, i_src_rep);// the whole matrix of context representation for every words in partial_sents is computed - which is also wasteful because we only need the representation of last comlumn?

	// only consider the prediction of last column in the matrix	
	dynet::Expression i_tgt_t;
	if (_decoder.get()->_batch_tlen == 1) i_tgt_t = i_tgt_ctx;
	else 
		//i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
		i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(_decoder.get()->_batch_tlen - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

	// output linear projections (w/ bias)
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
	dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

	// compute softmax prediction (note: return a batch of softmaxes)
	return dynet::softmax(i_r_t / sm_temp);// softmax w/ temperature
}

dynet::Expression TransformerContextModel::build_graph(dynet::ComputationGraph &cg
        , const WordIdSentences& ssents
        , const WordIdSentences& tsents
        , vector<vector<vector<dynet::real>>> wordrep_doc
        , vector<unsigned int> sids
        , ModelStats* pstats
        , bool is_eval_on_dev)
{
    // encode source
    dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, ssents, pstats);// ((num_units, Lx), batch_size)

    //combine the encoded source with the context
    i_src_ctx = _encoder_context->build_graph(cg, i_src_ctx, wordrep_doc, sids);// ((num_units, Lx), batch_size)
    
    // decode target
    dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, tsents, i_src_ctx);// ((num_units, Ly), batch_size)

    // get losses
    dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
    dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

// both of the followings work well!
#ifndef USE_LINEAR_TRANSFORMATION_BROADCASTING
    // Note: can be more efficient if using direct computing for i_tgt_ctx (e.g., use affine_transform)
	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t && pstats)
				pstats->_words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
			}
		}

		// compute the logit
		//dynet::Expression i_tgt_t = dynet::select_cols(i_tgt_ctx, {t});// shifted right
		dynet::Expression i_tgt_t = dynet::pick(i_tgt_ctx, t, 1);// shifted right, ((|V_T|, 1), batch_size)

		// output linear projections
		dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

		// log_softmax and loss
		dynet::Expression i_err;
		if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
		{// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
			// label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
			dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);
			dynet::Expression i_pre_loss = -dynet::pick(i_log_softmax, next_words);
			dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
			i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
		}
		else
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);

		v_errors.push_back(i_err);
	}
#else // Note: this way is much faster!
    // compute the logit and linear projections
    dynet::Expression i_r = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_ctx});// ((|V_T|, (Ly-1)), batch_size)

    std::vector<dynet::Expression> v_errors;
    unsigned tlen = _decoder.get()->_batch_tlen;
    std::vector<unsigned> next_words(tsents.size());
    for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
        for(size_t bs = 0; bs < tsents.size(); bs++){
            next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
            if (tsents[bs].size() > t && pstats) {
                pstats->_words_tgt++;
                if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
            }
        }

        // get the prediction at timestep t
        //dynet::Expression i_r_t = dynet::select_cols(i_r, {t});// shifted right, ((|V_T|, 1), batch_size)
        dynet::Expression i_r_t = dynet::pick(i_r, t, 1);// shifted right, ((|V_T|, 1), batch_size)

        // log_softmax and loss
        dynet::Expression i_err;
        if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
        {// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
            // label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
            dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);

            dynet::Expression i_pre_loss = -dynet::pick(i_log_softmax, next_words);
            dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
            i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
        }
        else
            i_err = dynet::pickneglogsoftmax(i_r_t, next_words);// ((1, 1), batch_size)

        v_errors.push_back(i_err);
    }
#endif

    dynet::Expression i_tloss = dynet::sum_batches(dynet::sum(v_errors));

    return i_tloss;
}

WordIdSentence TransformerContextModel::greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, unsigned length_ratio)
{
    //_tfc._is_training = false;

    const int& sos_sym = _tfc._sm._kTGT_SOS;
    const int& eos_sym = _tfc._sm._kTGT_EOS;

    // start of sentence
    WordIdSentence target;
    target.push_back(sos_sym);

    dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);// ToDo: batch decoding

    std::vector<dynet::Expression> aligns;// FIXME: unused
    unsigned t = 0;
    while (target.back() != eos_sym)
    {
        cg.checkpoint();

        dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, target, false, aligns);
        auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto pr_w = ydist[w];
        for (unsigned x = 1; x < ydist.size(); ++x) {
            if (ydist[x] > pr_w) {
                w = x;
                pr_w = ydist[w];
            }
        }

        // break potential infinite loop
        if (t > length_ratio * source.size()) {
            w = eos_sym;
            pr_w = ydist[w];
        }

        // Note: use pr_w if getting the probability of the generated sequence!

        target.push_back(w);
        t += 1;
        if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

        cg.revert();
    }

    cg.clear();

    //_tfc._is_training = true;
    return target;
}

WordIdSentence TransformerContextModel::greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, vector<vector<dynet::real>> wordrep_sent,
                                                      unsigned length_ratio)
{
    //_tfc._is_training = false;

    const int& sos_sym = _tfc._sm._kTGT_SOS;
    const int& eos_sym = _tfc._sm._kTGT_EOS;

    // start of sentence
    WordIdSentence target;
    target.push_back(sos_sym);

    dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source));// ToDo: batch decoding

    //combine the encoded source with the context
    i_src_rep = _encoder_context->build_graph(cg, i_src_rep, wordrep_sent);// ((num_units, Lx), batch_size)

    std::vector<dynet::Expression> aligns;// FIXME: unused
    unsigned t = 0;
    while (target.back() != eos_sym)
    {
        cg.checkpoint();

        dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, target, false, aligns);

        auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto pr_w = ydist[w];
        for (unsigned x = 1; x < ydist.size(); ++x) {
            if (ydist[x] > pr_w) {
                w = x;
                pr_w = ydist[w];
            }
        }

        // break potential infinite loop
        if (t > length_ratio * source.size()) {
            w = eos_sym;
            pr_w = ydist[w];
        }

        // Note: use pr_w if getting the probability of the generated sequence!

        target.push_back(w);
        t += 1;
        if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

        cg.revert();
    }

    cg.clear();

    //_tfc._is_training = true;
    return target;
}

WordIdSentence TransformerContextModel::greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, vector<vector<vector<dynet::real>>> wordrep_doc,
                                                      vector<unsigned int> sids, unsigned length_ratio)
{
    //_tfc._is_training = false;

    const int& sos_sym = _tfc._sm._kTGT_SOS;
    const int& eos_sym = _tfc._sm._kTGT_EOS;

    // start of sentence
    WordIdSentence target;
    target.push_back(sos_sym);

    dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source));// ToDo: batch decoding

    //combine the encoded source with the context
    i_src_rep = _encoder_context->build_graph(cg, i_src_rep, wordrep_doc, sids);// ((num_units, Lx), batch_size)
    
    std::vector<dynet::Expression> aligns;// FIXME: unused
    unsigned t = 0;
    while (target.back() != eos_sym)
    {
        cg.checkpoint();

        dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, target, false, aligns);

        auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto pr_w = ydist[w];
        for (unsigned x = 1; x < ydist.size(); ++x) {
            if (ydist[x] > pr_w) {
                w = x;
                pr_w = ydist[w];
            }
        }

        // break potential infinite loop
        if (t > length_ratio * source.size()) {
            w = eos_sym;
            pr_w = ydist[w];
        }

        // Note: use pr_w if getting the probability of the generated sequence!

        target.push_back(w);
        t += 1;
        if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

        cg.revert();
    }

    cg.clear();

    //_tfc._is_training = true;
    return target;
}

dynet::ParameterCollection& TransformerContextModel::get_model_parameters(){
	return *_all_params.get();
}

dynet::ParameterCollection& TransformerContextModel::get_context_model_parameters(){
    return *_context_params.get();
}

void TransformerContextModel::initialise_baseparams_from_file(const std::string &params_file)
{
	//dynet::load_dynet_model(params_file, _base_params.get());// FIXME: use binary streaming instead for saving disk spaces?
    TextFileLoader loader(params_file);
    loader.populate(*_base_params.get());
}

void TransformerContextModel::initialise_params_from_file(const std::string &params_file)
{
    //dynet::load_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
    TextFileLoader loader(params_file);
    loader.populate(*_all_params.get());
}

void TransformerContextModel::save_params_to_file(const std::string &params_file)
{
	//dynet::save_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
    TextFileSaver saver(params_file);
    saver.save(*_all_params.get());
}

void TransformerContextModel::set_dropout(bool is_activated){
	_tfc._use_dropout = is_activated;
}

dynet::Dict& TransformerContextModel::get_source_dict()
{
	return _dicts.first;
}
dynet::Dict& TransformerContextModel::get_target_dict()
{
	return _dicts.second;
}

TransformerConfig& TransformerContextModel::get_config(){
	return _tfc;
}

//---

}; // namespace transformer



