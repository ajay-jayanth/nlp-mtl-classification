import torch
from torch import nn
from fastai.vision.learner import create_body

class mtl_classifier(nn.Module):
    def __init__(self, body, source_head, audience_head, content_head, head_hyper_params):
        super(mtl_classifier, self).__init__()
        self.body = create_body(body)
        self.source_head = source_head(head_hyper_params['source'])
        self.audience_head = audience_head(head_hyper_params['audience'])
        self.content_head = content_head(head_hyper_params['content'])

    def forward(self, x):
        x = self.body(x)
        source_vals = self.source_head(x)
        audience_vals = self.audience_head(x)
        content_vals = self.content_head(x)

        source_label = torch.argmax(source_vals, 1)
        audience_label = torch.argmax(audience_vals, 1)
        content_label = torch.argmax(content_vals, 1)
        return [source_label, audience_label, content_label]

class mtl_loss_wrapper(nn.Module):
    def __init__(self, task_num, loss_func=nn.CrossEntropyLoss):
        super(mtl_loss_wrapper, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.loss_func = loss_func() # loss_func must be a classification-appropriate loss

    def forward(self, pred, target):
        losses = []
        for y_hat, y_test, precision, log_var in zip(pred.T, target.T, self.log_vars):
            base_loss = self.loss_func(y_hat, y_test)
            precision = torch.exp(log_var)
            weighted_loss = precision * base_loss + log_var
            losses.append(weighted_loss)

        return torch.sum(losses)
