import { Tooltip as TooltipPrimitive } from 'radix-ui'
import { cn } from '@/lib/utils'

function TooltipProvider({ delayDuration = 150, ...props }) {
    return <TooltipPrimitive.Provider delayDuration={delayDuration} {...props} />
}

function Tooltip({ ...props }) {
    return <TooltipPrimitive.Root data-slot="tooltip" {...props} />
}

function TooltipTrigger({ ...props }) {
    return <TooltipPrimitive.Trigger data-slot="tooltip-trigger" {...props} />
}

function TooltipContent({ className, sideOffset = 4, ...props }) {
    return (
        <TooltipPrimitive.Portal>
            <TooltipPrimitive.Content
                data-slot="tooltip-content"
                sideOffset={sideOffset}
                className={cn(
                    'z-50 overflow-hidden rounded-md bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md',
                    className,
                )}
                {...props}
            />
        </TooltipPrimitive.Portal>
    )
}

export { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent }
