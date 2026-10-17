"""Mini-batch orchestration, using only our engine for training gradients."""
import numpy as np
from nn.losses import cross_entropy
from prune.scores import magnitude, update_saliency
from prune.schedule import target_sparsity
from prune.sparse import dense_forward

def evaluate(model, inputs, targets):
    logits = dense_forward(model, inputs)
    return {'loss': float(cross_entropy(logits, targets).item()),
            'accuracy': float(np.mean(logits.argmax(axis=1) == targets))}

def train(model, optimizer, dataset, config, pruner=None):
    x, y = dataset['train']
    epochs, batch_size = config['epochs'], config['batch_size']
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    rng = np.random.default_rng(config.get('seed', 0))
    # Only weight matrices are prunable; biases keep their full shape and values.
    parameters = [p for p in model.parameters() if p.prunable]
    # Saliency starts at zero and accumulates across batches throughout training.
    scores = {p: np.zeros_like(p.data) for p in parameters}
    settings = config['pruning']
    warmup, ramp = settings['warmup_fraction'], settings['ramp_fraction']
    if warmup < 0 or ramp <= 0 or warmup+ramp > 1 or settings['interval_epochs'] < 1:
        raise ValueError("Invalid pruning schedule")
    start, end = epochs*warmup, epochs*(warmup+ramp)
    method = config.get('method', 'dense')
    if method not in ('dense','magnitude','saliency'):
        raise ValueError("Unknown importance method")
    if settings.get('regrowth', False):
        raise ValueError("Automatic regrowth is not implemented")
    history = []
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(x))
        for offset in range(0, len(x), batch_size):
            batch = order[offset:offset+batch_size]
            optimizer.zero_grad()
            loss = cross_entropy(model(x[batch]), y[batch])
            if not np.isfinite(loss.item()):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if method == 'saliency':
                # Read weight and gradient together, before Adam changes the weight.
                # This estimates the effect of deleting the weight seen by this batch.
                for p in parameters:
                    scores[p] = update_saliency(
                        scores[p], p.data, p.grad, settings['score_ema_beta']
                    )
            # Adam updates only after this batch's saliency has been measured.
            optimizer.step()
        # Pruning happens between epochs, after the batch gradients and Adam updates.
        if pruner is not None and (epoch % settings['interval_epochs']==0 or epoch==epochs):
            if method == 'magnitude':
                scores = {p: magnitude(p.data) for p in parameters}
            # During warm-up this is zero; it rises during the ramp, then stays fixed.
            scheduled = target_sparsity(epoch, start, end, config['target_sparsity'])
            pruner.update(model, optimizer, scores, scheduled)
        training = evaluate(model, *dataset['train'])
        validation = evaluate(model, *dataset['validation'])
        active = sum(int(p.mask.sum()) for p in parameters)
        total = sum(p.data.size for p in parameters)
        history.append({'epoch': epoch, 'training_loss': training['loss'],
                        'training_accuracy': training['accuracy'],
                        'validation_loss': validation['loss'],
                        'validation_accuracy': validation['accuracy'],
                        'active_weights': active, 'achieved_sparsity': 1-active/total})
    optimizer.zero_grad()
    return history
