def alloc_transform_helper(signals, transforms, sigs, sidx, RaggedArray,
        outsig_fn, insig_fn,
        ):
    # -- this is a function that is used to construct the
    #    so-called transforms and filters, which map from signals to signals.

    tidx = idxs(transforms)
    tf_by_outsig = defaultdict(list)
    for tf in transforms:
        tf_by_outsig[outsig_fn(tf)].append(tf)

    # N.B. the optimization below may not be valid
    # when alpha is not a scalar multiplier
    tf_weights = RaggedArray(
        [[tf.alpha] for tf in transforms])
    tf_Ns = [1] * len(transforms)
    tf_Ms = [1] * len(transforms)

    tf_sigs = sigs.shallow_copy()
    del sigs

    # -- which transform(s) does each signal use
    tf_weights_js = [[tidx[tf] for tf in tf_by_outsig[sig]]
        for sig in signals]

    # -- which corresponding(s) signal is transformed
    tf_signals_js = [[sidx[insig_fn(tf)] for tf in tf_by_outsig[sig]]
        for sig in signals]

    # -- Optimization:
    #    If any output signal is the sum of 
    #    consecutive weights with consecutive signals
    #    then turn that into *one* dot product
    #    of a new longer weight vector with a new
    #    longer signal vector.
    #    TODO: still do something like this for a 
    #    *part* of the wjs/sjs if possible
    #    TODO: sort the sjs or wjs to canonicalize things
    if 0:
        wstarts = []
        wlens = []
        sstarts = []
        slens = []
        for ii, (wjs, sjs) in enumerate(
                zip(tf_weights_js, tf_signals_js)):
            assert len(wjs) == len(sjs)
            K = len(wjs)
            if len(wjs) <= 1:
                continue
            if wjs != range(wjs[0], wjs[0] + len(wjs)):
                continue
            if sjs != range(sjs[0], sjs[0] + len(sjs)):
                continue
            # -- precondition satisfied
            # XXX length should be the sum of the lenghts
            #     of the tf_weights[i] for i in wjs
            wstarts.append(int(tf_weights.starts[wjs[0]]))
            wlens.append(K)
            new_w_j = len(tf_weights_js) + len(wstarts) - 1
            tf_Ns.append(K)
            tf_Ms.append(1)
            sstarts.append(int(tf_sigs.starts[sjs[0]]))
            slens.append(K)
            new_s_j = len(tf_signals_js) + len(sstarts) - 1
            tf_weights_js[ii] = [new_w_j]
            tf_signals_js[ii] = [new_s_j]

        tf_weights.add_views(wstarts, wlens)
        tf_sigs.add_views(sstarts, slens)

    if 0:
        for ii, (wjs, sjs) in enumerate(
                zip(tf_weights_js, tf_signals_js)):
            if wjs:
                print wjs, sjs

    return locals()



