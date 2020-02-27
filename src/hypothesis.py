# coding=utf8

import copy
from rule import define_rule


class ActionInfo(object):
    """sufficient statistics for making a prediction of an action at a time step"""

    def __init__(self, action=None):
        self.t = 0
        self.score = 0
        self.parent_t = -1
        self.action = action
        self.frontier_prod = None
        self.frontier_field = None

        # for GenToken actions only
        self.copy_from_src = False
        self.src_token_position = -1


class Hypothesis(object):
    def __init__(self, is_sketch=False):
        self.actions = []
        self.action_infos = []
        self.inputs = []
        self.score = 0.
        self.t = 0
        self.is_sketch = is_sketch
        self.sketch_step = 0
        self.sketch_attention_history = list()

    def get_availableClass(self):
        """
        return the available action class
        :return:
        """

        # TODO: it could be update by speed
        # return the available class using rule
        # FIXME: now should change for these 11: "Filter 1 ROOT",
        def check_type(lists):
            for s in lists:
                if type(s) == int:
                    return False
            return True

        stack = [define_rule.Root1]
        for action in self.actions:
            infer_action = action.get_next_action(is_sketch=self.is_sketch)
            infer_action.reverse()
            if stack[-1] is type(action):
                stack.pop()
                # check if the are non-terminal
                if check_type(infer_action):
                    stack.extend(infer_action)
            else:
                raise RuntimeError("Not the right action")

        result = stack[-1] if len(stack) > 0 else None

        return result

    @classmethod
    def get_parent_action(cls, actions):
        """

        :param actions:
        :return:
        """

        def check_type(lists):
            for s in lists:
                if type(s) == int:
                    return False
            return True

        # check the origin state Root
        if len(actions) == 0:
            return None

        stack = [define_rule.Root1]
        for id_x, action in enumerate(actions):
            infer_action = action.get_next_action()
            for ac in infer_action:
                ac.parent = action
                ac.pt = id_x
            infer_action.reverse()
            if stack[-1] is type(action):
                stack.pop()
                # check if the are non-terminal
                if check_type(infer_action):
                    stack.extend(infer_action)
            else:
                for t in actions:
                    if type(t) != define_rule.C:
                        print(t, end="")
                print('asd')
                print(action)
                print(stack[-1])
                raise RuntimeError("Not the right action")
        result = stack[-1] if len(stack) > 0 else None

        return result

    def apply_action(self, action):
        # TODO: not finish implement yet
        self.t += 1
        self.actions.append(action)

    def clone_and_apply_action(self, action):
        new_hyp = self.copy()
        new_hyp.apply_action(action)

        return new_hyp

    def clone_and_apply_action_info(self, action_info):
        action = action_info.action
        action.score = action_info.score
        new_hyp = self.clone_and_apply_action(action)
        new_hyp.action_infos.append(action_info)
        new_hyp.sketch_step = self.sketch_step
        new_hyp.sketch_attention_history = copy.copy(self.sketch_attention_history)

        return new_hyp

    def copy(self):
        new_hyp = Hypothesis(is_sketch=self.is_sketch)
        # if self.tree:
        #     new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.score = self.score
        new_hyp.t = self.t
        new_hyp.sketch_step = self.sketch_step
        new_hyp.sketch_attention_history = copy.copy(self.sketch_attention_history)

        return new_hyp

    def infer_n(self):
        if len(self.actions) > 4:
            prev_action = self.actions[-3]
            if isinstance(prev_action, define_rule.Filter):
                if prev_action.id_c > 11:
                    # Nested Query, only select 1 column
                    return ['N A']
            if self.actions[0].id_c != 3:
                return [self.actions[3].production]
        return define_rule.N._init_grammar()

    @property
    def completed(self):
        return True if self.get_availableClass() is None else False

    @property
    def is_valid(self):
        actions = self.actions
        return self.check_sel_valid(actions)

    def check_sel_valid(self, actions):
        find_sel = False
        sel_actions = list()
        for ac in actions:
            if type(ac) == define_rule.Sel:
                find_sel = True
            elif find_sel and type(ac) in [define_rule.N, define_rule.T, define_rule.C, define_rule.A]:
                if type(ac) not in [define_rule.N]:
                    sel_actions.append(ac)
            elif find_sel and type(ac) not in [define_rule.N, define_rule.T, define_rule.C, define_rule.A]:
                break

        if find_sel is False:
            return True

        # not the complete sel lf
        if len(sel_actions) % 3 != 0:
            return True

        sel_string = list()
        for i in range(len(sel_actions) // 3):
            if (sel_actions[i * 3 + 0].id_c, sel_actions[i * 3 + 1].id_c, sel_actions[i * 3 + 2].id_c) in sel_string:
                return False
            else:
                sel_string.append(
                    (sel_actions[i * 3 + 0].id_c, sel_actions[i * 3 + 1].id_c, sel_actions[i * 3 + 2].id_c))
        return True


if __name__ == '__main__':
    test = Hypothesis(is_sketch=True)
    # print(define_rule.Root1(1).get_next_action())
    test.actions.append(define_rule.Root1(3))
    test.actions.append(define_rule.Root(5))

    print(str(test.get_availableClass()))
    # print(Hypothesis.get_parent_action([define_rule.Root(3), define_rule.Sel(0), define_rule.N(1), define_rule.A(0),
    #                                     define_rule.A(0), define_rule.Filter(5)]).parent)

