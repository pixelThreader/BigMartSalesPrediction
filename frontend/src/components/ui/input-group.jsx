import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

function InputGroup({ className, ...props }) {
    return (
        <div
            data-slot="input-group"
            className={cn(
                'flex w-full items-stretch overflow-hidden rounded-md border border-input bg-background shadow-xs',
                className,
            )}
            {...props}
        />
    )
}

function InputGroupInput({ className, ...props }) {
    return (
        <Input
            data-slot="input-group-input"
            className={cn(
                'min-w-0 flex-1 rounded-none border-0 bg-transparent shadow-none focus-visible:z-10',
                className,
            )}
            {...props}
        />
    )
}

function InputGroupAddon({ className, align = 'inline-end', ...props }) {
    return (
        <div
            data-slot="input-group-addon"
            data-align={align}
            className={cn(
                'flex shrink-0 items-center border-input bg-muted/40 px-3 text-sm text-muted-foreground',
                align === 'inline-start'
                    ? 'order-first border-r'
                    : 'order-last border-l',
                className,
            )}
            {...props}
        />
    )
}

function InputGroupButton({ className, size = 'default', ...props }) {
    return (
        <Button
            data-slot="input-group-button"
            size={size}
            className={cn('rounded-none border-0 border-l border-input px-4 shadow-none', className)}
            {...props}
        />
    )
}

export {
    InputGroup,
    InputGroupAddon,
    InputGroupButton,
    InputGroupInput,
}
